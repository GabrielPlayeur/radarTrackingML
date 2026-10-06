# Pistage radar maritime : quelles briques d'un filtre de Kalman l'IA peut-elle améliorer ?

> **Projet en cours.**
> C'est un support d'apprentissage (radar, filtrage de Kalman, pistage), pas un produit :
> le code est écrit pour être lu et compris, et n'a aucune prétention opérationnelle.

## Objectif

Construire pas à pas une chaîne de pistage radar maritime, avec tout le filtrage codé à la main
(sans bibliothèque de filtrage dans `src/`), puis mesurer honnêtement ce qu'une brique d'IA
apporte, ou non, par rapport à un EKF et à un IMM.

## État d'avancement

| Jalon | Contenu | État |
|---|---|---|
| S0 | Bases radar + Kalman | en cours |
| A | Kalman / EKF, une cible | à faire |
| B | Chaîne signal : chirp → distance-Doppler → CFAR | à faire |
| C | Multi-cibles : GNN + clutter + IMM + OSPA | à faire |
| D | Une expérience IA (gain appris ou R adaptatif) | à faire |

Les résultats et figures seront ajoutés ici (figures dans `figures/`) à la fin de chaque jalon.

## Structure

```
src/
├── radar/      # forme d'onde, simulation, traitement (CFAR, Keystone)
├── sim/        # trajectoires, scénarios multi-cibles
├── tracking/   # kf, ekf, imm, gnn, gestion de pistes
├── learned/    # kalmannet, R adaptatif
└── metrics/    # rmse, ospa, nees
notebooks/      # un notebook par jalon
tests/          # validations théoriques (pytest)
cpp/            # jalon G (optionnel)
figures/        # figures générées pour ce README
```

## Installation

Python 3.11 et [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
uv run pytest
```

## Règles du projet

- `filterpy` n'est utilisé que dans `tests/`, comme référence de comparaison.
- Chaque validation théorique (pic au bon endroit, NEES, jacobienne, ...) est un test `pytest`.
- Graines aléatoires fixées, paramètres dans des dataclasses.
