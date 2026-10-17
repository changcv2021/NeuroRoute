"""Version3 confirmation identity over unchanged validated fast PPO algorithms.

Only complete trusted-local checkpoints of the same phase, protocol, selection,
method, seed, source and runtime can be restored. Development warm starts are
not supported. Validation is a distinct phase and never a confirmation result.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp

from .confirmation_protocol import ConfirmationConfig
from .fast_reproducible_training import FastReproduciblePPOTrainer
from .fast_centralized_training import FastCentralPPOTrainer
from .reproducible_training import canonical_arrays


class ConfirmationCheckpointMixin:
    def _bind_confirmation(self):
        self.algorithm_implementation_sha256 = self.implementation_sha256
        source = (self.algorithm_implementation_sha256.encode() + Path(__file__).read_bytes()
                  + Path(__file__).with_name("confirmation_protocol.py").read_bytes())
        self.implementation_sha256 = hashlib.sha256(source).hexdigest()

    def checkpoint_metadata(self):
        return {"format_version": 3, "checkpoint_kind": "neuroroute_confirmation_v1",
            "phase": self.config.phase, "training_seed": self.config.seed,
            "method": self.config.method, "method_key": self.config.method_key,
            "config": asdict(self.config), "protocol_sha256": self.config.protocol_sha256,
            "selection_sha256": self.config.selection_sha256,
            "resolved_protocol_sha256": self.config.protocol_sha256,
            "selected_config_sha256": self.config.selection_sha256,
            "training_environment_steps_target": self.config.num_updates*self.config.num_envs*self.config.horizon,
            "algorithm_implementation_sha256": self.algorithm_implementation_sha256,
            "implementation_sha256": self.implementation_sha256,
            "environment_class": type(self.env).__module__ + "." + type(self.env).__qualname__,
            "env_dt": float(self.env.dt), "numerical_runtime": self.runtime,
            "normalizer": {"kind": "fixed_identity", "updates": 0},
            "scheduler": {"kind": "constant_learning_rate", "value": self.config.learning_rate},
            "initialization": "fresh_seed_no_development_checkpoint",
            "communication_applicability": "not_applicable" if self.config.method == "B_CENTRAL" else "actual_flags_and_sent_64bit_packets"}

    def save_checkpoint(self, path, state):
        self.config.validate_manifests()
        if any(jax.core.get_aval(x).weak_type for x in jax.tree.leaves(state)):
            raise ValueError("Version3 requires strong explicit array dtypes")
        steps = int(state.environment_steps)
        expected = self.config.num_envs*self.config.horizon*int(state.update_index)
        if steps != expected or steps > self.config.num_envs*self.config.horizon*self.config.num_updates:
            raise ValueError("Checkpoint steps differ from the exact resolved update budget")
        content = {**self.checkpoint_metadata(), "state": jax.device_get(state)}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+".partial")
        with temporary.open("wb") as stream:
            pickle.dump(content, stream, protocol=5)
        temporary.replace(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load_checkpoint(self, path):
        self.config.validate_manifests()
        with Path(path).open("rb") as stream:
            content = pickle.load(stream)
        expected = self.checkpoint_metadata()
        mismatch = [k for k, v in expected.items() if content.get(k) != v]
        if mismatch:
            raise ValueError("Checkpoint version/phase/protocol/seed/source/runtime mismatch: " + ", ".join(mismatch))
        state = canonical_arrays(jax.tree.map(jnp.asarray, content["state"]))
        steps, updates = int(state.environment_steps), int(state.update_index)
        if (updates < 0 or updates > self.config.num_updates
                or steps != updates*self.config.num_envs*self.config.horizon):
            raise ValueError("Checkpoint counters violate the frozen step budget")
        return state

    def update(self, state):
        if int(state.update_index) >= self.config.num_updates:
            raise ValueError("Resolved training budget is already complete")
        return super().update(state)


class ConfirmationModularTrainer(ConfirmationCheckpointMixin, FastReproduciblePPOTrainer):
    def __init__(self, env, config, **kwargs):
        if not isinstance(config, ConfirmationConfig) or config.method == "B_CENTRAL":
            raise ValueError("Use a modular ConfirmationConfig")
        super().__init__(env, config, **kwargs)
        self._bind_confirmation()


class ConfirmationCentralTrainer(ConfirmationCheckpointMixin, FastCentralPPOTrainer):
    def __init__(self, env, config, **kwargs):
        if not isinstance(config, ConfirmationConfig) or config.method != "B_CENTRAL":
            raise ValueError("Use a central ConfirmationConfig")
        super().__init__(env, config, **kwargs)
        self._bind_confirmation()


def make_confirmation_trainer(env, config, **kwargs):
    return (ConfirmationCentralTrainer if config.method == "B_CENTRAL" else ConfirmationModularTrainer)(env, config, **kwargs)
