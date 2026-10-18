# Installation and usage

Use Python 3.12 and a suitable Slurm GPU allocation. Training, simulation, dependency installation, large data downloads, and GPU tests belong on compute nodes. The training entry point checks that it is running in a GPU allocation.

## Pinned environment

| Component | Version |
| --- | --- |
| JAX / JAXLIB / CUDA12 plugin / PJRT | 0.8.2 |
| MuJoCo / mujoco-mjx | 3.6.0 |
| Brax | 0.14.2 |
| Flax | 0.11.2 |
| Optax | 0.2.6 |
| MuJoCo Playground | `4057c147714b6ac09b395377f1a1724bbeacc4d3` |
| MuJoCo Menagerie | `1b86ece576591213e2b666ebf59508454200ca97` |

The complete dependency lock and package hashes are in `manifests/requirements_jax082.lock`. Set `NR_WORK` to a persistent data volume and run the following inside an allocation with a supported Python 3.12 interpreter:

```bash
export NR_CODE="$PWD"
export NR_WORK="/your/persistent/work/NeuroRoute"
mkdir -p "$NR_WORK/software" "$NR_WORK/cache"
python3.12 -m venv "$NR_WORK/env"
export NR_PY="$NR_WORK/env/bin/python"
export PIP_CACHE_DIR="$NR_WORK/cache/pip"
export UV_CACHE_DIR="$NR_WORK/cache/uv"
"$NR_PY" -m pip install 'uv==0.12.13'

git clone --filter=blob:none https://github.com/google-deepmind/mujoco_playground.git \
  "$NR_WORK/software/mujoco_playground"
git -C "$NR_WORK/software/mujoco_playground" checkout --detach \
  4057c147714b6ac09b395377f1a1724bbeacc4d3
mkdir -p "$NR_WORK/software/mujoco_playground/mujoco_playground/external_deps"
git clone --filter=blob:none https://github.com/google-deepmind/mujoco_menagerie.git \
  "$NR_WORK/software/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie"
git -C "$NR_WORK/software/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie" \
  checkout --detach 1b86ece576591213e2b666ebf59508454200ca97

"$NR_WORK/env/bin/uv" pip sync --python "$NR_PY" --require-hashes \
  "$NR_CODE/manifests/requirements_jax082.lock"
"$NR_PY" -m pip install --no-deps --no-build-isolation \
  -e "$NR_WORK/software/mujoco_playground" -e "$NR_CODE"
"$NR_PY" -m pip check
```

All training and evaluation modules are installed within `neuroroute`. Set these numerical runtime options before importing JAX:

```bash
export MUJOCO_GL=egl
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_enable_scatter_determinism_expander=true"
```

The training script supplies these defaults when unset and checks the required XLA flags. Driver and GPU compatibility must match the pinned runtime.

## Training

```bash
"$NR_PY" "$NR_CODE/scripts/train_flycns.py" --budget 0.25 --variant full --seed 101 \
  --output "$NR_WORK/outputs/flycns_b025_seed101" --evaluate-final
```

The output directory must not already exist. The script saves its configuration, progress, and policy checkpoints. `--evaluate-final` runs fixed command evaluations after training. `--save-training-state` also saves optimizer and environment state. This entry point starts fresh runs; the trainer exposes restoration callbacks for applications that implement their own continuation workflow.

Default PPO settings are 8192 environments, unroll length 20, batch size 256, 32 minibatches, four updates per batch, learning rate 0.0003, and a request for 200M environment steps. The trainer rounds to complete update blocks. Independent seeds can run concurrently on separate GPUs.

Use `--budget 0.75`, `0.5`, or `0.25`. The `full` variant is the final model. Component ablations are `no_bio`, `no_curriculum`, `no_age`, and `no_opt`; all retain receiver memory. Settings and the directional prior are in `configs/flycns.json`.

## Evaluation and connectome inputs

`configs/behavior.json` specifies eight reset seeds and two evaluation modes: the environment's commands and a scripted sequence. The script uses four 250-step blocks: forward motion, lateral motion, yaw rotation, and forward motion with reverse yaw. Evaluation computes tracking, survival, and logical communication cost. Rendering functions generate videos and plots from recorded trajectories.

Connectome input locations, sizes, and checksums are in `manifests/connectome_manifest.json`; extraction rules are in `configs/connectome_banc_v888.json`.

```bash
"$NR_PY" scripts/fetch_connectome.py --data-dir "$NR_WORK/data/banc"
"$NR_PY" -m neuroroute.connectome.extract --help
"$NR_PY" -m neuroroute.connectome.derive_prior --help
```

Raw datasets, derived matrices, weights, and generated experiment outputs are not bundled. Store generated outputs outside the repository or in ignored output directories.

## Checks

`python3 tools/check_release.py` uses only the standard library to check the file inventory, source hashes, syntax, and excluded output types. Hashes are recorded in `manifests/source.json`.

Run `python -m pytest tests` in the installed compute environment for model, environment, and connectome tests.
