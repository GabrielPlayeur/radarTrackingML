"""Validations de l'évaluation : métriques, IC, références, pipeline complet sur de minuscules jeux (jalon D)."""

import json

import numpy as np
import pytest
import torch

from learned.dataset import DatasetParams
from learned.evaluate import (
    BURN_IN,
    evaluate,
    fixed_gain_model,
    format_tables,
    load_models,
    main,
    rmse_with_ci,
    seed_ci,
    squared_position_error,
    subset_sums,
    when_it_fails,
)
from learned.kalmannet import KalmanNet, KalmanNetParams
from learned.train import TrainConfig, train

NET = KalmanNetParams()
TINY = DatasetParams(n_train=16, n_val=8, n_test=8, n_steps=30, seed=1)


def make_checkpoint(tmp_path, seed=0, data=TINY):
    """Entraîne 2 époques sur un jeu minuscule pour obtenir un vrai point de contrôle (pas un résultat)."""
    config = TrainConfig(epochs=2, batch_size=8, seed=seed, out_dir=str(tmp_path))
    train(config, data, NET, log=lambda *_: None)
    return str(tmp_path / f"kalmannet_seed{seed}_best.pt")


def test_squared_position_error_ignores_velocity():
    """(x − x̂)² + (y − ŷ)² : 3-4-5, la vitesse n'intervient pas."""
    truth = np.zeros((1, 2, 4))
    est = np.array([[[3.0, 4.0, 99.0, 99.0], [0.0, 0.0, 5.0, 5.0]]])
    np.testing.assert_allclose(squared_position_error(truth, est), [[25.0, 0.0]])


def test_subset_sums_drop_burn_in_and_respect_the_mask():
    """Seuls les instants k ≥ BURN_IN du masque comptent."""
    sq = np.ones((2, 30))
    mask = np.zeros((2, 30), dtype=bool)
    mask[0, :] = True
    se, count = subset_sums(sq, mask)
    np.testing.assert_array_equal(count, [30 - BURN_IN, 0])
    np.testing.assert_array_equal(se, [30 - BURN_IN, 0])


def test_rmse_and_bootstrap_ci_on_known_data():
    """Erreur constante de 5 m : RMSE = 5 et IC nul ; erreurs variables : RMSE exact et IC qui rétrécit avec N."""
    rng = np.random.default_rng(0)
    assert rmse_with_ci((np.full(50, 25.0 * 20), np.full(50, 20)), rng) == pytest.approx((5.0, 0.0))
    errors = np.random.default_rng(1).normal(0, 10, (400, 20)) ** 2
    small = rmse_with_ci((errors[:40].sum(1), np.full(40, 20)), rng)
    large = rmse_with_ci((errors.sum(1), np.full(400, 20)), rng)
    assert large[0] == pytest.approx(np.sqrt(errors.mean()))
    assert large[1] < 0.5 * small[1]


def test_rmse_of_empty_subset_is_nan():
    """Un sous-ensemble sans instant (ex. : virage dans un jeu sans virage) donne NaN, pas une erreur."""
    rmse, ci = rmse_with_ci((np.zeros(5), np.zeros(5, dtype=int)), np.random.default_rng(0))
    assert np.isnan(rmse) and np.isnan(ci)


def test_seed_ci_matches_student_formula():
    """[1, 2, 3] : moyenne 2, demi-IC = t(0,975 ; 2 ddl)·σ/√3 = 4,303/√3 ; une seule graine : IC indéfini."""
    mean, half = seed_ci([1.0, 2.0, 3.0])
    assert mean == 2.0 and half == pytest.approx(4.302653 / np.sqrt(3), rel=1e-5)
    assert np.isnan(seed_ci([4.0])[1])


def test_fixed_gain_model_has_a_constant_prior_gain():
    """Tête annulée : le gain vaut l'a priori alpha-bêta, quelles que soient les entrées."""
    model = fixed_gain_model(NET)
    gains = model.gain_from_output(model.head(torch.randn(10, NET.hidden_gru))).view(10, 8)
    expected = torch.tensor([NET.gain_prior_pos, 0, 0, NET.gain_prior_pos,
                             NET.gain_prior_vel, 0, 0, NET.gain_prior_vel], dtype=torch.float32)
    torch.testing.assert_close(gains, expected.expand(10, 8))


def test_references_only_pipeline_and_expected_ordering():
    """Sans point de contrôle : 4 conditions × 3 références, valeurs finies ; l'IMM bat l'EKF-CV en virage fort."""
    outcome = evaluate([], n_test=8)
    assert set(outcome["results"]) == {"nominal", "strong_turns", "noisy", "long"}
    for methods in outcome["results"].values():
        assert set(methods) == {"EKF-CV", "IMM (CV + CT±)", "gain fixe (non appris)"}
        assert all(np.isfinite(m["global"]["rmse"]) for m in methods.values())
    strong = outcome["results"]["strong_turns"]
    assert strong["IMM (CV + CT±)"]["virage"]["rmse"] < strong["EKF-CV"]["virage"]["rmse"]
    assert all(t > 0 for t in outcome["timings"].values())
    assert "KalmanNet" not in outcome["timings"]
    assert when_it_fails(outcome["results"]) == ["(aucun point de contrôle fourni : pas de KalmanNet à comparer)"]


def test_full_pipeline_with_checkpoints_tables_and_figure(tmp_path):
    """Avec 2 points de contrôle : ligne KalmanNet avec IC sur les graines, rapport Markdown, JSON et figure écrits."""
    paths = [make_checkpoint(tmp_path / f"s{i}", seed=i) for i in range(2)]
    out_json, fig = tmp_path / "res" / "D.json", tmp_path / "fig.png"
    main(["--checkpoints", *paths, "--n-test", "6", "--out", str(out_json), "--figure", str(fig)])
    data = json.loads(out_json.read_text())
    km = data["results"]["nominal"]["KalmanNet"]["global"]
    assert len(km["per_seed"]) == 2 and np.isfinite(km["rmse"]) and data["n_seeds"] == 2
    assert data["n_params"] == KalmanNet(NET).n_parameters < 50_000
    report = out_json.with_suffix(".md").read_text()
    assert "### Quand ça ne marche pas" in report and "| KalmanNet |" in report and "paramètres appris" in report
    assert fig.exists() and fig.stat().st_size > 5_000


def test_load_models_rejects_checkpoints_with_different_data(tmp_path):
    """Deux points de contrôle entraînés sur des données différentes ne sont pas comparables : erreur explicite."""
    a = make_checkpoint(tmp_path / "a")
    b = make_checkpoint(tmp_path / "b", data=DatasetParams(n_train=16, n_val=8, n_test=8, n_steps=30, seed=2))
    with pytest.raises(ValueError):
        load_models([a, b])


def test_format_tables_marks_empty_subsets():
    """Un sous-ensemble sans donnée s'affiche « — » dans le tableau."""
    nan = {"rmse": float("nan"), "ci": float("nan")}
    ok = {"rmse": 30.0, "ci": 1.0}
    results = {c: {"EKF-CV": {"global": ok, "virage": nan, "ligne_droite": ok}} for c in
               ("nominal", "strong_turns", "noisy", "long")}
    text = format_tables(results, {"EKF-CV": 40.0}, 19592, 0)
    assert "| EKF-CV | 30.0 ± 1.0 | — | 30.0 ± 1.0 |" in text


def test_footnote_mentions_seeds_only_when_there_are_some():
    """Sans point de contrôle, le pied de tableau ne parle pas de « 0 graines »."""
    nan = {"rmse": 30.0, "ci": 1.0}
    one = {c: {"EKF-CV": {"global": nan, "virage": nan, "ligne_droite": nan}} for c in
           ("nominal", "strong_turns", "noisy", "long")}
    assert "graines" not in format_tables(one, {"EKF-CV": 1.0}, 19592, 0)
    assert "3 graines" in format_tables(one, {"EKF-CV": 1.0}, 19592, 3)
