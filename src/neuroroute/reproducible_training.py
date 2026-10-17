"""Versioned deterministic-runtime candidate; legacy checkpoints stay immutable.

This changes no PPO/reward/packet objective. It canonicalizes JAX input types
and binds complete checkpoints to the exact numerical runtime. Acceptance is
conditional on actual GPU repeat/resume tests, not on setting flags alone.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import jaxlib
import mujoco

from .training import PPOTrainer, implementation_fingerprint


def canonical_arrays(tree):
    return jax.tree.map(lambda value: jax.lax.convert_element_type(value, value.dtype), tree)


def numerical_runtime():
    flags = dict(token.removeprefix("--").split("=", 1)
                 for token in os.environ.get("XLA_FLAGS", "").split() if "=" in token)
    if flags.get("xla_gpu_deterministic_ops") != "true" or flags.get("xla_gpu_enable_scatter_determinism_expander") != "false":
        raise ValueError("This candidate requires both recorded deterministic/scatter-workaround flags")
    if str(jax.config.jax_default_matmul_precision) != "highest":
        raise ValueError("Expected the predeclared highest matrix multiplication precision")
    return {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "mujoco": mujoco.__version__,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "matmul_precision": str(jax.config.jax_default_matmul_precision),
            "x64": bool(jax.config.jax_enable_x64),
            "devices": [{"platform": d.platform, "kind": d.device_kind} for d in jax.devices()],
            "input_array_typing": "all_strong_explicit_dtype"}


class ReproduciblePPOTrainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime = numerical_runtime()
        source = implementation_fingerprint().encode() + Path(__file__).read_bytes()
        self.implementation_sha256 = hashlib.sha256(
            source + json.dumps(self.runtime, sort_keys=True).encode()).hexdigest()

    def _initialize(self, key):
        return canonical_arrays(super()._initialize(key))

    def _rollout_step(self, state, unused):
        state, transition = super()._rollout_step(state, unused)
        return canonical_arrays(state), canonical_arrays(transition)

    def _update(self, state):
        return canonical_arrays(super()._update(state))

    def save_checkpoint(self, path, state):
        if any(jax.core.get_aval(x).weak_type for x in jax.tree.leaves(state)):
            raise ValueError("Noncanonical state: refusing to discard weak array metadata")
        content = {"format_version": 2, "config": asdict(self.config), "env_dt": float(self.env.dt),
                   "implementation_sha256": self.implementation_sha256,
                   "environment_class": type(self.env).__module__ + "." + type(self.env).__qualname__,
                   "numerical_runtime": self.runtime,
                   "normalizer": {"kind": "fixed_identity", "updates": 0},
                   "scheduler": {"kind": "constant_learning_rate", "value": self.config.learning_rate},
                   "state": jax.device_get(state)}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as stream:
            pickle.dump(content, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load_checkpoint(self, path):
        with Path(path).open("rb") as stream:
            content = pickle.load(stream)
        expected = {"format_version": 2, "config": asdict(self.config), "env_dt": float(self.env.dt),
                    "implementation_sha256": self.implementation_sha256,
                    "environment_class": type(self.env).__module__ + "." + type(self.env).__qualname__,
                    "numerical_runtime": self.runtime,
                    "normalizer": {"kind": "fixed_identity", "updates": 0},
                    "scheduler": {"kind": "constant_learning_rate", "value": self.config.learning_rate}}
        if any(content.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint/runtime/config mismatch; legacy v1 is not silently migrated")
        return canonical_arrays(jax.tree.map(jnp.asarray, content["state"]))
