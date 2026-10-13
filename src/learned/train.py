"""Entraînement de KalmanNet : script en ligne de commande, journaux de perte et points de contrôle.

Exemple (une graine) :
    uv run python -m learned.train --seed 0 --out-dir checkpoints
Estimation du temps sans entraîner (quelques lots chronométrés) :
    uv run python -m learned.train --estimate
La perte est l'erreur quadratique moyenne d'état (position / vitesse normalisées) de x̂_{k|k} sur toute la séquence,
rétropropagée à travers les pas de filtre (le gain dépend des estimations passées).
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from learned.dataset import DatasetParams, make_splits, to_tensors
from learned.kalmannet import KalmanNet, KalmanNetParams


@dataclass
class TrainConfig:
    """Réglages de l'entraînement (les autres paramètres viennent de DatasetParams et KalmanNetParams)."""

    seed: int = 0  # graine du réseau (initialisation) et de l'ordre des lots
    epochs: int = 60
    batch_size: int = 64
    lr: float = 2e-3
    grad_clip: float = 1.0  # norme maximale du gradient
    burn_in: int = 2  # instants initiaux ignorés dans la perte (initialisation sans vitesse)
    patience: int = 15  # arrêt anticipé : époques sans amélioration de la validation
    device: str = "cpu"  # « cpu » ou « cuda » (un GRU de 64 unités est plus rapide sur CPU)
    out_dir: str = "checkpoints"


def state_loss(estimates: torch.Tensor, truth: torch.Tensor, params: KalmanNetParams, burn_in: int) -> torch.Tensor:
    """Perte : moyenne de ‖Δposition‖²/pos_scale² + ‖Δvitesse‖²/vel_scale² sur les instants k ≥ burn_in."""
    err = (estimates - truth)[:, burn_in:]
    return ((err[..., :2] / params.pos_scale) ** 2).sum(-1).mean() + ((err[..., 2:] / params.vel_scale) ** 2).sum(-1).mean()


def position_rmse(estimates: torch.Tensor, truth: torch.Tensor, burn_in: int) -> float:
    """RMSE de position en mètres sur les instants k ≥ burn_in."""
    err = (estimates - truth)[:, burn_in:, :2]
    return float(err.pow(2).sum(-1).mean().sqrt())


@torch.no_grad()
def validate(model: KalmanNet, truth: torch.Tensor, z: torch.Tensor, config: TrainConfig) -> tuple:
    """(perte, RMSE de position [m]) sur le jeu de validation, sans gradient."""
    model.eval()
    est = model(z)
    return float(state_loss(est, truth, model.params, config.burn_in)), position_rmse(est, truth, config.burn_in)


def train(config: TrainConfig, data_params: DatasetParams, net_params: KalmanNetParams = KalmanNetParams(),
          splits: dict | None = None, log=print) -> dict:
    """Entraîne KalmanNet ; écrit le meilleur point de contrôle (validation), le dernier et un journal CSV.

    Retourne l'historique {epoch, train_loss, val_loss, val_rmse}. `splits` permet de passer des données déjà
    générées (tests) ; sinon elles sont produites par make_splits(data_params), identiques pour toutes les graines.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    splits = splits or make_splits(data_params)
    device = torch.device(config.device)
    train_truth, train_z = to_tensors(splits["train"], device)
    val_truth, val_z = to_tensors(splits["val"], device)

    model = KalmanNet(net_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, config.patience // 3))
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"kalmannet_seed{config.seed}"

    def save(path, epoch, val_loss):
        torch.save({"state_dict": model.state_dict(), "net_params": asdict(net_params),
                    "data_params": asdict(data_params), "config": asdict(config),
                    "epoch": epoch, "val_loss": val_loss}, path)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_rmse": []}
    best, stale, n = float("inf"), 0, len(train_z)
    with open(f"{stem}_log.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_rmse_m", "lr", "seconds"])
        start = time.perf_counter()
        for epoch in range(1, config.epochs + 1):
            model.train()
            order = rng.permutation(n)
            losses = []
            for i in range(0, n, config.batch_size):
                idx = torch.as_tensor(order[i : i + config.batch_size], device=device)
                loss = state_loss(model(train_z[idx]), train_truth[idx], net_params, config.burn_in)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach()))
            val_loss, val_rmse = validate(model, val_truth, val_z, config)
            scheduler.step(val_loss)
            train_loss = float(np.mean(losses))
            for key, value in zip(history, (epoch, train_loss, val_loss, val_rmse)):
                history[key].append(value)
            seconds = time.perf_counter() - start
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_rmse:.3f}",
                             f"{optimizer.param_groups[0]['lr']:.2e}", f"{seconds:.1f}"])
            handle.flush()
            log(f"époque {epoch:3d}/{config.epochs}  perte train {train_loss:.4f}  val {val_loss:.4f}  "
                f"RMSE val {val_rmse:5.1f} m  ({seconds:.0f} s)")
            save(f"{stem}_last.pt", epoch, val_loss)
            if val_loss < best:
                best, stale = val_loss, 0
                save(f"{stem}_best.pt", epoch, val_loss)
            else:
                stale += 1
                if stale >= config.patience:
                    log(f"arrêt anticipé : pas d'amélioration depuis {config.patience} époques")
                    break
    return history


def estimate_time(config: TrainConfig, data_params: DatasetParams, net_params: KalmanNetParams, n_batches: int = 4) -> dict:
    """Chronomètre quelques lots (aller-retour complet) et extrapole la durée d'une graine, sans entraîner.

    Les lots sont mesurés sur des trajectoires aléatoires : aucun point de contrôle n'est écrit et le modèle est jeté.
    """
    torch.manual_seed(0)
    device = torch.device(config.device)
    model = KalmanNet(net_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    z = torch.randn(config.batch_size, data_params.n_steps, 2, device=device) * 0.01
    z[..., 0] = z[..., 0] * 1e5 + 5000.0
    truth = torch.zeros(config.batch_size, data_params.n_steps, 4, device=device)
    times = []
    for _ in range(n_batches + 1):  # le premier lot sert d'échauffement
        start = time.perf_counter()
        loss = state_loss(model(z), truth, net_params, config.burn_in)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    per_batch = float(np.mean(times[1:]))
    batches_per_epoch = int(np.ceil(data_params.n_train / config.batch_size))
    per_epoch = per_batch * batches_per_epoch * 1.08  # + validation (~8 %)
    return {"seconds_per_batch": per_batch, "batches_per_epoch": batches_per_epoch,
            "seconds_per_epoch": per_epoch, "seconds_total_max": per_epoch * config.epochs}


def build_parser() -> argparse.ArgumentParser:
    """Arguments : un par champ de TrainConfig, plus la taille des jeux de données (DatasetParams)."""
    parser = argparse.ArgumentParser(description="Entraîne KalmanNet (gain de Kalman appris par un GRU).")
    defaults, data_defaults = TrainConfig(), DatasetParams()
    for f in fields(TrainConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(getattr(defaults, f.name)),
                            default=getattr(defaults, f.name))
    parser.add_argument("--n-train", type=int, default=data_defaults.n_train)
    parser.add_argument("--n-val", type=int, default=data_defaults.n_val)
    parser.add_argument("--n-steps", type=int, default=data_defaults.n_steps)
    parser.add_argument("--data-seed", type=int, default=data_defaults.seed,
                        help="graine des données (commune à toutes les graines d'entraînement)")
    parser.add_argument("--estimate", action="store_true", help="estime la durée sans entraîner, puis quitte")
    return parser


def main(argv=None) -> None:
    """Point d'entrée : lit les arguments, estime ou entraîne."""
    args = build_parser().parse_args(argv)
    config = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    data_params = DatasetParams(n_train=args.n_train, n_val=args.n_val, n_steps=args.n_steps, seed=args.data_seed)
    net_params = KalmanNetParams(dt=data_params.dt, sigma_r=data_params.sigma_r, sigma_theta=data_params.sigma_theta)
    n_params = KalmanNet(net_params).n_parameters
    print(f"KalmanNet : {n_params} paramètres ; {data_params.n_train} trajectoires × {data_params.n_steps} pas ; "
          f"appareil : {config.device}")
    if args.estimate:
        info = estimate_time(config, data_params, net_params)
        print(json.dumps(info, indent=2))
        print(f"≈ {info['seconds_per_epoch']:.0f} s par époque, au plus {info['seconds_total_max'] / 60:.0f} min par graine "
              f"({config.epochs} époques, moins avec l'arrêt anticipé)")
        return
    history = train(config, data_params, net_params)
    print(f"terminé : meilleure RMSE de validation {min(history['val_rmse']):.1f} m ; "
          f"points de contrôle dans {config.out_dir}/")


if __name__ == "__main__":
    main()
