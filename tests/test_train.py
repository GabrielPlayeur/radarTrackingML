"""Validations de la boucle d'entraînement (sur de minuscules jeux : ce ne sont pas des entraînements réels)."""

import csv
from dataclasses import asdict

import numpy as np
import pytest
import torch

from learned.dataset import DatasetParams, make_splits
from learned.kalmannet import KalmanNet, KalmanNetParams
from learned.train import TrainConfig, build_parser, estimate_time, main, position_rmse, state_loss, train

NET = KalmanNetParams()
TINY = DatasetParams(n_train=16, n_val=8, n_test=8, n_steps=30, seed=1)
SPLITS = make_splits(TINY)


def quiet_train(tmp_path, **overrides):
    """Entraînement minuscule (8 trajectoires par lot, quelques époques) dans un dossier temporaire."""
    config = TrainConfig(**{"epochs": 3, "batch_size": 8, "out_dir": str(tmp_path), **overrides})
    return train(config, TINY, NET, SPLITS, log=lambda *_: None), config


def test_state_loss_is_zero_for_perfect_estimates_and_normalised():
    """Perte nulle si l'estimation est exacte ; 50 m de position (pos_scale) ou 1 m/s (vel_scale) valent 1 chacun."""
    truth = torch.zeros(2, 10, 4)
    assert state_loss(truth, truth, NET, burn_in=2) == 0
    pos = truth.clone()
    pos[..., 0] = NET.pos_scale
    assert state_loss(pos, truth, NET, 2) == pytest.approx(1.0)
    vel = truth.clone()
    vel[..., 3] = NET.vel_scale
    assert state_loss(vel, truth, NET, 2) == pytest.approx(1.0)


def test_burn_in_steps_are_ignored_in_the_loss_and_the_rmse():
    """Une erreur sur les 2 premiers instants n'entre ni dans la perte ni dans le RMSE (burn_in = 2)."""
    truth = torch.zeros(1, 10, 4)
    est = truth.clone()
    est[:, :2, :] = 1e4
    assert state_loss(est, truth, NET, burn_in=2) == 0
    assert position_rmse(est, truth, burn_in=2) == 0
    assert position_rmse(est, truth, burn_in=0) > 0


def test_training_writes_checkpoints_and_log(tmp_path):
    """Un entraînement écrit le meilleur point, le dernier, le journal CSV, et l'historique est cohérent."""
    history, config = quiet_train(tmp_path)
    assert len(history["epoch"]) == 3 and all(np.isfinite(history["val_loss"]))
    for suffix in ("best.pt", "last.pt", "log.csv"):
        assert (tmp_path / f"kalmannet_seed0_{suffix}").exists()
    rows = list(csv.reader(open(tmp_path / "kalmannet_seed0_log.csv")))
    assert rows[0][:4] == ["epoch", "train_loss", "val_loss", "val_rmse_m"] and len(rows) == 4


def test_checkpoint_round_trip_reproduces_the_model(tmp_path):
    """Le point de contrôle contient les paramètres du réseau et sa configuration : le recharger redonne le même modèle."""
    quiet_train(tmp_path)
    ckpt = torch.load(tmp_path / "kalmannet_seed0_last.pt", weights_only=False)
    assert ckpt["epoch"] == 3 and ckpt["net_params"] == asdict(NET) and ckpt["config"]["seed"] == 0
    z = torch.as_tensor(SPLITS["test"].z, dtype=torch.float32)
    models = []
    for _ in range(2):
        model = KalmanNet(KalmanNetParams(**ckpt["net_params"]))
        model.load_state_dict(ckpt["state_dict"])
        models.append(model.eval())
    torch.testing.assert_close(models[0](z), models[1](z))
    assert not torch.equal(models[0](z), KalmanNet(NET).eval()(z))  # différent d'un réseau non entraîné


def test_same_seed_same_history_different_seed_differs(tmp_path):
    """Entraînement reproductible par graine (CPU) ; une autre graine change l'initialisation et l'ordre des lots."""
    a, _ = quiet_train(tmp_path / "a", seed=3)
    b, _ = quiet_train(tmp_path / "b", seed=3)
    c, _ = quiet_train(tmp_path / "c", seed=4)
    assert a["train_loss"] == b["train_loss"]
    assert a["train_loss"] != c["train_loss"]


def test_optimisation_reduces_the_loss_on_a_tiny_set(tmp_path):
    """Sur 8 trajectoires, 40 époques divisent la perte par plus de 2 d'entraînement : le gradient apprend vraiment."""
    history, _ = quiet_train(tmp_path, epochs=40, lr=5e-3, batch_size=8, patience=100)
    assert history["train_loss"][-1] < 0.5 * history["train_loss"][0], history["train_loss"][::10]


def test_early_stopping_when_validation_stalls(tmp_path):
    """Taux d'apprentissage nul : la validation ne bouge pas, l'entraînement s'arrête après patience + 1 époques."""
    history, _ = quiet_train(tmp_path, epochs=50, lr=0.0, patience=2)
    assert len(history["epoch"]) == 3


def test_time_estimate_is_positive_and_scales_with_epochs():
    """L'estimation de durée est positive et proportionnelle au nombre d'époques."""
    config = TrainConfig(epochs=10, batch_size=8)
    small = DatasetParams(n_train=16, n_steps=30)
    info = estimate_time(config, small, NET, n_batches=2)
    assert info["seconds_per_batch"] > 0 and info["batches_per_epoch"] == 2
    assert info["seconds_total_max"] == pytest.approx(10 * info["seconds_per_epoch"])


def test_cli_defaults_and_estimate_mode(capsys):
    """La CLI expose chaque réglage ; --estimate affiche une durée et n'écrit aucun point de contrôle."""
    args = build_parser().parse_args([])
    assert args.epochs == TrainConfig().epochs and args.n_train == DatasetParams().n_train and not args.estimate
    main(["--estimate", "--n-train", "16", "--n-steps", "20", "--batch-size", "8", "--epochs", "5"])
    out = capsys.readouterr().out
    assert "paramètres" in out and "s par époque" in out
