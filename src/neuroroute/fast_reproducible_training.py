"""Isolated JAX 0.8.2 fast-scatter candidate, sharing unchanged canonical PPO.

Checkpoint format 2 still describes the same complete state schema. Its
implementation/runtime fingerprint is distinct from the slow 0.6.2 runtime;
neither trainer accepts the other's checkpoints.
"""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import jax

from .training import PPOTrainer, implementation_fingerprint
from .reproducible_training import ReproduciblePPOTrainer


def fast_numerical_runtime():
    flags = dict(token.removeprefix("--").split("=", 1)
                 for token in os.environ.get("XLA_FLAGS", "").split() if "=" in token)
    if flags.get("xla_gpu_deterministic_ops") != "true" or flags.get("xla_gpu_enable_scatter_determinism_expander") != "true":
        raise ValueError("The fast-scatter candidate requires both explicit true flags")
    packages = {p: version(p) for p in (
        "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "mujoco", "mujoco-mjx",
        "flax", "brax", "optax", "orbax-checkpoint", "numpy", "scipy", "ml-dtypes")}
    if any(packages[p] != "0.8.2" for p in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt")):
        raise ValueError("This versioned candidate requires the complete JAX 0.8.2 CUDA12 stack")
    if str(jax.config.jax_default_matmul_precision) != "highest":
        raise ValueError("Expected highest matrix multiplication precision")
    devices = jax.devices()
    if not devices or any(d.platform != "gpu" for d in devices):
        raise ValueError("Candidate acceptance is specific to GPU execution")
    return {"runtime_candidate": "jax082_fast_scatter", "jax": jax.__version__,
            "packages": packages, "python": platform.python_version(),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "matmul_precision": str(jax.config.jax_default_matmul_precision),
            "x64": bool(jax.config.jax_enable_x64),
            "devices": [{"platform": d.platform, "kind": d.device_kind,
                         "platform_version": d.client.platform_version} for d in devices],
            "input_array_typing": "all_strong_explicit_dtype"}


class FastReproduciblePPOTrainer(ReproduciblePPOTrainer):
    def __init__(self, *args, **kwargs):
        # Reuse the canonical overrides, while binding to this distinct runtime
        # instead of constructing the slow runtime's strict metadata contract.
        PPOTrainer.__init__(self, *args, **kwargs)
        self.runtime = fast_numerical_runtime()
        source = (implementation_fingerprint().encode()
                  + Path(__file__).with_name("reproducible_training.py").read_bytes()
                  + Path(__file__).read_bytes())
        self.implementation_sha256 = hashlib.sha256(
            source + json.dumps(self.runtime, sort_keys=True).encode()).hexdigest()
