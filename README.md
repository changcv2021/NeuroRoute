# FlyCNS: Connectome-Grounded Information Organization for Communication-Constrained Embodied Control

<a href="https://arxiv.org/abs/2609.28816">
  <img src="https://img.shields.io/badge/READ%20PAPER-arXiv%202609.28816-DC2626?style=for-the-badge&amp;labelColor=991B1B" height="38" alt="Read Paper">
</a>

Code for communication-constrained locomotion on the Go1 quadruped, built with JAX, MuJoCo MJX, and Brax PPO. The repository includes FlyCNS-v2, ASYM, and PERIODIC policy implementations, along with connectome prior extraction, communication accounting, and behavioral evaluation.

This release contains source code, configuration, dependency specifications, and usage documentation. Experimental results, reports, figures, videos, model weights, and run records are kept in the local research archive.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/neuroroute/` | Environments, policies, message transport, communication metrics, and connectome processing |
| `recovery_v2/` | FlyCNS-v2 PPO, message-state replay, losses, and behavioral evaluation |
| `recovery_v1/` | Shared utilities used by the current training and evaluation code |
| `configs/` | Network, training, ablation, and evaluation input configurations |
| `scripts/train_flycns.py` | Public entry point for a fresh training run |
| `scripts/fetch_connectome.py` | Download pinned versions of connectome input data |
| `tests/` | Unit test source for communication, policies, environments, and connectome processing |
| `manifests/` | Dependency lock file and external data download manifest |
| `release/source_snapshot.json` | SHA256 manifest of preserved source files |

See the [module guide](docs/CODE_MAP.md) and [environment and usage guide](docs/USAGE.md) for details. These supplementary guides are currently in Chinese.

## Getting started

Use Python 3.12. The following lightweight commands inspect the source and configuration without installing the scientific computing dependencies:

```bash
python3 tools/check_release.py
python3 scripts/train_flycns.py --help
python3 scripts/train_flycns.py --budget 0.25 --variant full --seed 101 --print-config
```

After setting up the dependencies as described in the usage guide, launch an independent training run inside a suitable Slurm allocation with one GPU:

```bash
python3 scripts/train_flycns.py --budget 0.25 --variant full --seed 101 \
  --output /your/work/outputs/flycns_b025_seed101 --evaluate-final
```

Supported target budgets are `0.75`, `0.5`, and `0.25`. Available variants are `full`, `no_bio`, `no_curriculum`, `no_age`, `no_opt`, and `bio_v1`. The communication budget is a training target; actual logical transmission cost is calculated from packet send records.

The public entry point reuses the preserved model and training implementation, starts from random initialization, and does not depend on approval records from historical local experiments. It was added for this release and has passed source integrity, static syntax, and CLI parsing checks; GPU training has not been rerun through this entry point. Earlier implementations remain available as library modules. Historical job schedulers and run records are outside the scope of this release.

See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) for third-party components. A repository-wide license has not yet been assigned to the original project code.
