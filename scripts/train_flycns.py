#!/usr/bin/env python3
"""Fresh FlyCNS training using the preserved model and PPO implementation."""
import argparse
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flycns.json")
    parser.add_argument("--budget", type=float, choices=(0.75, 0.5, 0.25), default=0.75)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--evaluate-final", action="store_true")
    parser.add_argument("--save-training-state", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.variant not in config["variants"]:
        parser.error("--variant must be one of: " + ", ".join(config["variants"]))
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    selected = dict(config, budget=args.budget, variant=args.variant, seed=args.seed)
    if args.print_config:
        print(json.dumps(selected, indent=2))
        return
    if args.output is None:
        parser.error("--output is required for training")
    if args.output.exists():
        parser.error("--output must be a new directory")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    os.environ.setdefault("JAX_ENABLE_X64", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    deterministic_flags = (
        "--xla_gpu_deterministic_ops=true",
        "--xla_gpu_enable_scatter_determinism_expander=true",
    )
    os.environ.setdefault("XLA_FLAGS", " ".join(deterministic_flags))
    if any(flag not in os.environ["XLA_FLAGS"].split() for flag in deterministic_flags):
        raise RuntimeError("Set the deterministic XLA_FLAGS documented in docs/USAGE.md")

    from neuroroute.runtime import require_compute, write_json, file_sha256
    require_compute(gpu=True)
    import jax
    import numpy as np
    from brax.training.acme import running_statistics
    from mujoco_playground import wrapper
    from mujoco_playground.config import locomotion_params
    from neuroroute.checkpoints import save_checkpoint
    from neuroroute.environment import FlyCNSGo1
    from neuroroute.training import ppo as trainer
    from neuroroute.models.probability import make_components, make_networks

    if jax.local_device_count() != 1 or jax.process_count() != 1:
        raise RuntimeError("This entry point requires one GPU per independent training run")
    runtime = {name: importlib.metadata.version(name) for name in config["runtime"]}
    if runtime != config["runtime"]:
        raise RuntimeError("Installed package versions differ from configs/flycns.json")
    env = FlyCNSGo1(args.budget)
    upstream = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain", impl="jax").to_dict()
    factory = upstream.pop("network_factory")
    canonical = lambda value: json.loads(json.dumps(value, sort_keys=True))
    if (canonical(upstream) != config["training"]
            or canonical(factory) != config["network_factory"]
            or canonical(env._config.to_dict()) != config["environment"]):
        raise RuntimeError("Upstream task/PPO/environment configuration differs from the source snapshot")
    kwargs = dict(
        mapping=env.leg_map.manifest(), architecture=config["architecture"],
        budget=args.budget, variant=config["variants"][args.variant],
        pi_bio=config["pi_bio"], bio_coefficient=config["bio_coefficient"], **factory,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", selected)
    contract = dict(family="FlyCNS-v2", entry="public_fresh_training", configuration=selected,
                    config_sha256=file_sha256(args.config), factory=kwargs)
    started = time.monotonic()

    def progress(steps, metrics):
        values = {name: float(np.asarray(value)) for name, value in metrics.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite training metric")
        row = dict(environment_steps=int(steps), elapsed_seconds=time.monotonic() - started,
                   metrics=values)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def checkpoint(steps, make_policy, params):
        save_checkpoint(args.output / "checkpoints" / f"{int(steps):012d}.pkl",
                        params, contract, steps)

    def training_state(steps, state):
        if not args.save_training_state:
            return
        folder = args.output / "training_states"
        folder.mkdir(exist_ok=True)
        destination = folder / f"{int(steps):012d}.pkl"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            pickle.dump(jax.device_get(state), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
        write_json(destination.with_suffix(".json"), dict(
            environment_steps=int(steps), sha256=file_sha256(destination), contract=contract))

    _, params, _ = trainer.train(
        environment=env, **config["training"], seed=args.seed,
        network_factory=functools.partial(make_networks, **kwargs),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress, policy_params_fn=checkpoint, training_state_fn=training_state,
    )
    if args.evaluate_final:
        from neuroroute.evaluation.behavior import make_rollout, evaluate
        protocol = json.loads((ROOT / "configs/behavior.json").read_text())
        network, diagnostic = make_components(
            env.observation_size, env.action_size,
            preprocess_observations_fn=running_statistics.normalize, **kwargs,
        )
        rollout = make_rollout(env, network, protocol, diagnostic)
        evaluate(rollout, params, env, protocol, args.output / "evaluation")


if __name__ == "__main__":
    main()
