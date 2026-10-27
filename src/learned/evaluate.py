"""Évaluation de KalmanNet contre l'EKF-CV, l'IMM et un gain fixe, sur le jeu de test et hors distribution.

Exemple (après entraînement des 3 graines) :
    uv run python -m learned.evaluate --checkpoints checkpoints/kalmannet_seed{0,1,2}_best.pt \\
        --out results/D_evaluation.json --figure figures/D_comparaison.png
Sans --checkpoints, seules les références (EKF-CV, IMM, gain fixe) sont évaluées.

Méthodes comparées (mêmes mesures, même initialisation depuis la première mesure) :
- EKF-CV : filtre de Kalman étendu, modèle CV, Q et R nominaux (jalon A) ;
- IMM (CV + CT±) : jalon C ;
- gain fixe : KalmanNet **non entraîné**, c.-à-d. son a priori alpha-bêta constant. C'est la référence qui isole
  ce que l'apprentissage apporte : gagner contre un EKF mal réglé ne prouve rien si un gain constant en fait autant ;
- KalmanNet : réseau entraîné, une valeur par graine d'entraînement.
Tous les filtres gardent leurs réglages nominaux dans les conditions hors distribution (ils ne savent pas que ça change).
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t

from learned.dataset import STRESS_KINDS, DatasetParams, make_splits, make_stress_test_set, to_tensors
from learned.kalmannet import KalmanNet, KalmanNetFilter, KalmanNetParams
from tracking.imm import make_imm_factory
from tracking.kf import run_filter
from tracking.track_manager import make_ekf_factory

SIGMA_A, OMEGA = 0.05, 0.02  # réglages nominaux des filtres (σ_a de la vérité, taux de virage des modèles CT)
BURN_IN = 10  # instants initiaux ignorés (transitoire d'initialisation)
CONDITIONS = {"nominal": "test (distribution d'entraînement)", "strong_turns": "virages 3 à 4 fois plus forts",
              "noisy": "bruit de mesure ×2", "long": "séquences 4 fois plus longues"}
REFERENCES = ["EKF-CV", "IMM (CV + CT±)", "gain fixe (non appris)"]


def squared_position_error(truth: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    """Erreur de position au carré (N, T) : (x − x̂)² + (y − ŷ)²."""
    return ((truth[..., :2] - estimates[..., :2]) ** 2).sum(-1)


def subset_sums(sq_err: np.ndarray, mask: np.ndarray) -> tuple:
    """Par trajectoire : somme des erreurs² et nombre d'instants retenus (masque ∧ k ≥ BURN_IN)."""
    keep = mask.copy()
    keep[:, :BURN_IN] = False
    return (sq_err * keep).sum(1), keep.sum(1)


def rmse_with_ci(sums: tuple, rng: np.random.Generator, n_boot: int = 1000) -> tuple:
    """RMSE (m) et demi-largeur de l'IC à 95 % obtenu en rééchantillonnant les trajectoires (bootstrap).

    RMSE = √(Σ erreurs² / Σ instants) sur les trajectoires tirées avec remise ; NaN si le sous-ensemble est vide.
    """
    se, count = sums
    if count.sum() == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(se), (n_boot, len(se)))
    boot = np.sqrt(se[idx].sum(1) / np.maximum(count[idx].sum(1), 1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.sqrt(se.sum() / count.sum())), float((hi - lo) / 2)


def seed_ci(values: list, level: float = 0.95) -> tuple:
    """Moyenne et demi-largeur de l'IC de Student sur les graines d'entraînement (NaN pour une seule graine)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(student_t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def run_numpy_filter(factory, z: np.ndarray) -> np.ndarray:
    """Estimations (N, T, 4) d'un filtre numpy (EKF, IMM) initialisé par la première mesure de chaque trajectoire."""
    return np.array([run_filter(factory(traj[0]), traj).x for traj in z])


@torch.no_grad()
def run_kalmannet(model: KalmanNet, z: np.ndarray) -> np.ndarray:
    """Estimations (N, T, 4) d'un KalmanNet sur toutes les trajectoires à la fois."""
    return model.eval()(torch.as_tensor(z, dtype=torch.float32)).numpy()


def fixed_gain_model(params: KalmanNetParams) -> KalmanNet:
    """KalmanNet dont la tête est annulée : gain constant égal à l'a priori alpha-bêta (aucun apprentissage)."""
    model = KalmanNet(params)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    return model


def load_models(paths: list) -> tuple:
    """Charge des points de contrôle ; retourne (modèles, KalmanNetParams, DatasetParams) en vérifiant leur cohérence."""
    ckpts = [torch.load(p, weights_only=False, map_location="cpu") for p in paths]
    net_params = {tuple(sorted(c["net_params"].items())) for c in ckpts}
    data_params = {json.dumps(c["data_params"], sort_keys=True) for c in ckpts}
    if len(net_params) > 1 or len(data_params) > 1:
        raise ValueError("les points de contrôle n'ont pas les mêmes paramètres de réseau ou de données")
    models = []
    for c in ckpts:
        model = KalmanNet(KalmanNetParams(**c["net_params"]))
        model.load_state_dict(c["state_dict"])
        models.append(model.eval())
    return models, KalmanNetParams(**ckpts[0]["net_params"]), DatasetParams(**{
        k: tuple(v) if isinstance(v, list) else v for k, v in ckpts[0]["data_params"].items()})


def test_batches(data_params: DatasetParams) -> dict:
    """Jeux de test : « nominal » (découpage test) et les trois conditions hors distribution."""
    batches = {"nominal": make_splits(data_params)["test"]}
    for kind in STRESS_KINDS:
        batches[kind] = make_stress_test_set(data_params, kind)
    return batches


def evaluate_condition(batch, data_params: DatasetParams, trained: list, net_params: KalmanNetParams,
                       rng: np.random.Generator) -> dict:
    """Métriques de toutes les méthodes sur un jeu de test : RMSE globale / en virage / en ligne droite (m, IC 95 %)."""
    args = (data_params.dt, SIGMA_A, data_params.sigma_r, data_params.sigma_theta)
    estimates = {
        "EKF-CV": [run_numpy_filter(make_ekf_factory(*args), batch.z)],
        "IMM (CV + CT±)": [run_numpy_filter(make_imm_factory(*args, turn_rates=(OMEGA, -OMEGA)), batch.z)],
        "gain fixe (non appris)": [run_kalmannet(fixed_gain_model(net_params), batch.z)],
    }
    if trained:
        estimates["KalmanNet"] = [run_kalmannet(m, batch.z) for m in trained]

    masks = {"global": np.ones_like(batch.in_turn), "virage": batch.in_turn, "ligne_droite": ~batch.in_turn}
    out = {}
    for name, runs in estimates.items():
        out[name] = {}
        for subset, mask in masks.items():
            per_run = [rmse_with_ci(subset_sums(squared_position_error(batch.truth, est), mask), rng) for est in runs]
            if name == "KalmanNet":  # une valeur par graine ; IC de Student sur les graines
                mean, half = seed_ci([r for r, _ in per_run])
                out[name][subset] = {"rmse": mean, "ci": half, "per_seed": [r for r, _ in per_run]}
            else:
                out[name][subset] = {"rmse": per_run[0][0], "ci": per_run[0][1]}
    return out


def time_per_update(data_params: DatasetParams, net_params: KalmanNetParams, trained: list, n_traj: int = 8) -> dict:
    """Temps moyen par mise à jour [µs] d'une cible, mesuré filtre par filtre, un seul fil d'exécution, CPU.

    Pour les filtres numpy : predict + update ; pour KalmanNet : un pas torch (batch de 1, sans gradient).
    """
    batch = make_splits(replace(data_params, n_train=1, n_val=1, n_test=n_traj))["test"]
    args = (data_params.dt, SIGMA_A, data_params.sigma_r, data_params.sigma_theta)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    timings = {}

    def clock(make_filter, step):
        """Durée moyenne de step(filtre, mesure) sur toutes les mesures k ≥ 1 ; make_filter(trajectoire) → filtre."""
        total, count = 0.0, 0
        for traj in batch.z:
            filt = make_filter(traj)
            for z in traj[1:]:
                start = time.perf_counter()
                step(filt, z)
                total += time.perf_counter() - start
                count += 1
        return 1e6 * total / count

    def numpy_step(filt, z):
        filt.predict()
        filt.update(z)

    ekf, imm = make_ekf_factory(*args), make_imm_factory(*args, turn_rates=(OMEGA, -OMEGA))
    timings["EKF-CV"] = clock(lambda traj: ekf(traj[0]), numpy_step)
    timings["IMM (CV + CT±)"] = clock(lambda traj: imm(traj[0]), numpy_step)
    fixed = fixed_gain_model(net_params)
    timings["gain fixe (non appris)"] = clock(lambda t: KalmanNetFilter(fixed, t[0]), numpy_step)
    if trained:
        timings["KalmanNet"] = clock(lambda t: KalmanNetFilter(trained[0], t[0]), numpy_step)
    torch.set_num_threads(previous)
    return timings


def format_tables(results: dict, timings: dict, n_params: int, n_seeds: int) -> str:
    """Rapport Markdown : un tableau par condition, puis le coût (paramètres, temps), puis « quand ça ne marche pas »."""
    lines = []

    def cell(entry):
        return "—" if np.isnan(entry["rmse"]) else (
            f"{entry['rmse']:.1f}" if np.isnan(entry["ci"]) else f"{entry['rmse']:.1f} ± {entry['ci']:.1f}")

    for cond, title in CONDITIONS.items():
        lines += [f"### {title}", "", "| méthode | RMSE globale [m] | RMSE en virage [m] | RMSE en ligne droite [m] |",
                  "|---|---|---|---|"]
        lines += [f"| {name} | {cell(m['global'])} | {cell(m['virage'])} | {cell(m['ligne_droite'])} |"
                  for name, m in results[cond].items()]
        lines.append("")
    note = "IC à 95 % : rééchantillonnage (bootstrap) des trajectoires de test"
    note += f" pour les références ; Student sur les {n_seeds} graines d'entraînement pour KalmanNet." if n_seeds else "."
    lines += [note, ""]
    lines += ["### Coût", "", "| méthode | paramètres appris | temps par mise à jour [µs] |", "|---|---|---|"]
    learned = {"KalmanNet": n_params}
    lines += [f"| {name} | {learned.get(name, 0)} | {t:.0f} |" for name, t in timings.items()]
    lines += ["", "### Quand ça ne marche pas", ""]
    lines += when_it_fails(results)
    return "\n".join(lines)


def when_it_fails(results: dict) -> list:
    """Constats automatiques : dans quelles conditions KalmanNet est moins bon que chaque référence (RMSE globale)."""
    if "KalmanNet" not in results["nominal"]:
        return ["(aucun point de contrôle fourni : pas de KalmanNet à comparer)"]
    notes = []
    for cond, title in CONDITIONS.items():
        kn = results[cond]["KalmanNet"]["global"]["rmse"]
        worse = [f"{ref} ({results[cond][ref]['global']['rmse']:.1f} m)"
                 for ref in REFERENCES if kn > results[cond][ref]["global"]["rmse"]]
        verdict = ("moins bon que " + ", ".join(worse)) if worse else "meilleur que toutes les références"
        notes.append(f"- **{title}** : KalmanNet {kn:.1f} m, {verdict}.")
    return notes


def plot_figure(results: dict, path: str) -> None:
    """Histogrammes de la RMSE globale par condition et par méthode (barres fines, valeur au sommet)."""
    import matplotlib.pyplot as plt

    colors = {"EKF-CV": "#2a78d6", "IMM (CV + CT±)": "#eb6834", "KalmanNet": "#1baf7a", "gain fixe (non appris)": "#898781"}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
                         "axes.grid": True, "grid.color": "#e1e0d9", "axes.axisbelow": True, "legend.frameon": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    methods = [m for m in colors if m in results["nominal"]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / len(methods)
    top = 0.0
    for i, name in enumerate(methods):
        x = np.arange(len(CONDITIONS)) + (i - (len(methods) - 1) / 2) * width
        values = [results[c][name]["global"]["rmse"] for c in CONDITIONS]
        errors = [0 if np.isnan(results[c][name]["global"]["ci"]) else results[c][name]["global"]["ci"] for c in CONDITIONS]
        ax.bar(x, values, width * 0.9, color=colors[name], label=name, yerr=errors, error_kw={"ecolor": "#52514e", "lw": 1})
        for xi, v, e in zip(x, values, errors):
            ax.text(xi, v + e + 1, f"{v:.0f}", ha="center", color="#52514e", fontsize=8)
            top = max(top, v + e)
    ax.set_ylim(0, 1.2 * top)  # place pour la légende au-dessus de la plus haute barre
    ax.set_xticks(np.arange(len(CONDITIONS)), list(CONDITIONS.values()), fontsize=9)
    ax.set_ylabel("RMSE de position [m]")
    ax.set_title("KalmanNet, EKF-CV, IMM et gain fixe : RMSE globale par condition", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=len(methods))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def evaluate(checkpoints: list, n_test: int | None = None, seed: int = 0) -> dict:
    """Évaluation complète : retourne {"results", "timings", "n_params", "n_seeds", "data_params"}."""
    if checkpoints:
        trained, net_params, data_params = load_models(checkpoints)
    else:
        trained, data_params = [], DatasetParams()
        net_params = KalmanNetParams(dt=data_params.dt, sigma_r=data_params.sigma_r, sigma_theta=data_params.sigma_theta)
    if n_test is not None:
        data_params = replace(data_params, n_test=n_test)
    rng = np.random.default_rng(seed)
    results = {cond: evaluate_condition(batch, data_params, trained, net_params, rng)
               for cond, batch in test_batches(data_params).items()}
    return {"results": results, "timings": time_per_update(data_params, net_params, trained),
            "n_params": trained[0].n_parameters if trained else KalmanNet(net_params).n_parameters,
            "n_seeds": len(trained), "data_params": asdict(data_params)}


def main(argv=None) -> None:
    """Point d'entrée : évalue, affiche le rapport Markdown, écrit le JSON et la figure."""
    parser = argparse.ArgumentParser(description="Compare KalmanNet, EKF-CV, IMM et gain fixe.")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="points de contrôle (une graine chacun)")
    parser.add_argument("--n-test", type=int, default=None, help="trajectoires par jeu de test (défaut : celui des données)")
    parser.add_argument("--seed", type=int, default=0, help="graine du bootstrap")
    parser.add_argument("--out", default="results/D_evaluation.json")
    parser.add_argument("--figure", default=None, help="chemin de la figure PNG (optionnel)")
    args = parser.parse_args(argv)
    outcome = evaluate(args.checkpoints, args.n_test, args.seed)
    report = format_tables(outcome["results"], outcome["timings"], outcome["n_params"], outcome["n_seeds"])
    print(report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(outcome, indent=2))
    Path(args.out).with_suffix(".md").write_text(report + "\n")
    if args.figure:
        plot_figure(outcome["results"], args.figure)


if __name__ == "__main__":
    main()
