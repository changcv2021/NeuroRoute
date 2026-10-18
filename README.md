# FlyCNS: Connectome-Grounded Information Organization for Communication-Constrained Embodied Control

<a href="https://arxiv.org/abs/2609.28816">
  <img src="https://img.shields.io/badge/READ%20PAPER-arXiv%202609.28816-DC2626?style=for-the-badge&amp;labelColor=991B1B" height="38" alt="Read Paper">
</a>

The final FlyCNS-v2 implementation for communication-constrained locomotion on the Go1 quadruped, built with JAX, MuJoCo MJX, and Brax PPO. FlyCNS combines causal communication gates, receiver message memory, budget curricula, and a connectome-derived directional prior.

This repository contains source code, configuration, dependencies, and usage documentation. Experimental results, reports, figures, videos, weights, and run records remain in the local research archive.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/neuroroute/models/` | Final actor, receiver memory, routing distributions, and shared neural layers |
| `src/neuroroute/training/` | PPO, budget curriculum, rollout collection, and losses |
| `src/neuroroute/evaluation/` | Command evaluation, metrics, and trajectory rendering |
| `src/neuroroute/connectome/` | Connectome processing and prior extraction |
| `src/neuroroute/environment.py` | Go1 environment and observation/action interfaces |
| `configs/` | Network, training, ablation, and evaluation inputs |
| `scripts/` | Training entry point and connectome data acquisition |
| `tests/` | Model, environment, and connectome tests |
| `manifests/` | Dependencies, external data locations, and source hashes |

See the [code guide](docs/CODE_MAP.md) and [installation and usage guide](docs/USAGE.md).

## Getting started

Use Python 3.12. These lightweight commands inspect the source and configuration without installing scientific computing dependencies:

```bash
python3 tools/check_release.py
python3 scripts/train_flycns.py --help
python3 scripts/train_flycns.py --budget 0.25 --variant full --seed 101 --print-config
```

After installing dependencies, launch an independent training run inside a suitable Slurm allocation with one GPU:

```bash
python3 scripts/train_flycns.py --budget 0.25 --variant full --seed 101 \
  --output /your/work/outputs/flycns_b025_seed101 --evaluate-final
```

Supported target budgets are `0.75`, `0.5`, and `0.25`. Use `full` for the final model. The same implementation supports component ablations through `no_bio`, `no_curriculum`, `no_age`, and `no_opt`. All variants use receiver memory. The budget is a training target; actual logical transmission cost is calculated from packet send records.

The training entry point starts from random initialization.

See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) for third-party components. A repository-wide license has not yet been assigned to the original project code.
