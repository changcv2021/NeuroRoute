"""Isolated actor-precision candidate; inherits frozen full-prefix PPO equations.

Only the actor compilation boundary and versioned checkpoint contract differ.
No checkpoint migration is performed, even though parameter shapes match v1.
"""

from dataclasses import asdict
import hashlib
from pathlib import Path
import pickle

import jax
import numpy as np

from .fast_reproducible_training import fast_numerical_runtime
from .published_training import canonical_arrays
from .recurrent_published_training import (
    RecurrentMatchConfig, RecurrentMatchTrainer as FrozenTrainer,
    implementation_fingerprint as frozen_fingerprint, load_recurrent_config)
from .recurrent_published_policy_v3 import RecurrentPublishedPolicy


CHECKPOINT_FORMAT = "neuroroute.match.recursive_fullprefix.actor_float64_boundary_float32.v3"


def implementation_fingerprint():
    digest = hashlib.sha256(frozen_fingerprint().encode())
    for name in ("recurrent_published_policy_v3.py", "recurrent_published_training_v3.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


class RecurrentMatchTrainer(FrozenTrainer):
    def __init__(self, env, config, **kwargs):
        super().__init__(env, config, **kwargs)
        # Base constructor creates lazy jit wrappers; no tracing/initialization
        # occurs before this replacement, so every path uses this same actor.
        self.actor = RecurrentPublishedPolicy(config.width, config.heads, config.layers)
        self.implementation_sha256 = implementation_fingerprint()

    def _checkpoint_contract(self):
        return {"format": CHECKPOINT_FORMAT, "config": asdict(self.config),
            "runtime": self.runtime, "environment_signature": self.environment_signature,
            "implementation_sha256": self.implementation_sha256,
            "prefix_forward": "complete_current_parameters_from_reset",
            "prefix_gradient": self.config.prefix_gradient,
            "numeric_boundary": "float64_internal_actor_float32_output_and_state"}

    def save_checkpoint(self, path, state):
        if self.runtime != fast_numerical_runtime():
            raise ValueError("Runtime changed after initialization")
        if any(bool(getattr(x, "weak_type", False)) for x in jax.tree.leaves(state)):
            raise ValueError("Weakly typed recurrent state cannot be saved")
        payload = {**self._checkpoint_contract(), "state": jax.device_get(state)}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_checkpoint(self, path):
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        expected = self._checkpoint_contract()
        if self.runtime != fast_numerical_runtime() or any(payload.get(k) != v for k, v in expected.items()):
            raise ValueError("Recursive MATCH v3 checkpoint/runtime mismatch; legacy formats are not migrated")
        state = canonical_arrays(payload["state"])
        lengths = np.asarray(state.episode_steps)
        if np.any(lengths < 0) or np.any(lengths >= self.config.episode_length):
            raise ValueError("Invalid reset-to-current prefix length")
        return state
