"""Strict version3 protocol for the paper-based recursive MATCH/PPO port."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

CONFIRMATION_SEEDS = (11, 22, 33, 44, 55)
VALIDATION_SEEDS = (907001, 907002)


def read_hashed(path, expected):
    path = Path(path)
    if (not isinstance(expected, str) or len(expected) != 64 or
            any(c not in "0123456789abcdef" for c in expected) or
            hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise ValueError("Required immutable manifest SHA256 mismatch")
    return json.loads(path.read_text())


@dataclass(frozen=True)
class PublishedConfirmationConfig:
    phase: str
    seed: int
    method_key: str
    method: str
    protocol_path: str
    protocol_sha256: str
    selection_path: str
    selection_sha256: str
    algorithm: dict

    def __getattr__(self, name):
        # Frozen PPO only reads algorithm fields. The actual phase and seed stay
        # visible here; no development seed or checkpoint is used to initialize.
        algorithm = object.__getattribute__(self, "algorithm")
        if name in algorithm:
            return algorithm[name]
        raise AttributeError(name)

    def validate_manifests(self):
        expected_seeds = CONFIRMATION_SEEDS if self.phase == "confirmation" else VALIDATION_SEEDS if self.phase == "validation" else ()
        if type(self.seed) is not int or self.seed not in expected_seeds:
            raise ValueError("Explicit confirmation seeds 11/22/33/44/55 or separate validation seeds 907001/907002 required")
        protocol = read_hashed(self.protocol_path, self.protocol_sha256)
        selection = read_hashed(self.selection_path, self.selection_sha256)
        if (protocol.get("schema_version") != 1 or protocol.get("phase") != self.phase
                or protocol.get("status") != f"resolved_{self.phase}"
                or protocol.get("training_seeds") != list(expected_seeds)
                or protocol.get("runtime_candidate") != "jax082_fast_scatter"
                or protocol.get("selected_methods", {}).get(self.method_key) != self.selection_sha256):
            raise ValueError("Protocol phase/seeds/runtime/selected-method hashes are not completely resolved")
        steps = protocol.get("training_environment_steps")
        if type(steps) is not int or steps <= 0:
            raise ValueError("Formal training_environment_steps is unresolved; no default budget is permitted")
        if (selection.get("schema_version") != 1 or selection.get("phase") != self.phase
                or selection.get("status") != "frozen" or selection.get("method") != self.method
                or selection.get("method_key") != self.method_key):
            raise ValueError("Frozen method selection does not match this run")
        selected = selection.get("hyperparameters")
        if not isinstance(selected, dict) or set(selected) & {"phase", "seed", "method", "num_updates"}:
            raise ValueError("Selected hyperparameters must exclude phase/seed/method/num_updates")
        actual = {k: v for k, v in self.algorithm.items() if k != "num_updates"}
        # JSON canonicalization equates a decoded list with the source tuple.
        if json.dumps(actual, sort_keys=True) != json.dumps(selected, sort_keys=True):
            raise ValueError("Run algorithm differs from the frozen selected hyperparameters")
        if any(type(actual.get(k)) is not int or actual[k] <= 0 for k in ("num_envs", "horizon")):
            raise ValueError("Selected rollout dimensions must be positive integers")
        per_update = actual["num_envs"]*actual["horizon"]
        if steps % per_update or self.algorithm.get("num_updates") != steps//per_update:
            raise ValueError("A shared exact step budget must be divisible by this method's rollout size")
        if self.phase == "confirmation":
            producer = selection.get("hpo_selection")
            if not isinstance(producer, dict):
                raise ValueError("Confirmation selection requires the frozen HPO producer artifact")
            hpo = read_hashed(producer["path"], producer["sha256"])
            selected_lr = hpo.get("selected", {}).get("learning_rate")
            if (hpo.get("status") != "selected_from_complete_frozen_development_matrix"
                    or hpo.get("formal_result") is not False
                    or hpo.get("family") != "published"
                    or not hpo.get("all_seed_method_values")
                    or selected_lr is None or selected_lr != actual.get("learning_rate") or not hpo.get("plan_sha256")):
                raise ValueError("Derived selection must use the already selected HPO learning rate and plan hash")
        return protocol, selection

    def validate(self):
        self.validate_manifests()
        if any(isinstance(v, float) and not math.isfinite(v) for v in self.algorithm.values()):
            raise ValueError("All selected numeric hyperparameters must be finite")
        for key in ("num_envs", "horizon", "num_updates", "epochs", "minibatch_size", "history_length", "episode_length"):
            if type(self.algorithm.get(key)) is not int or self.algorithm[key] <= 0:
                raise ValueError("All selected count/step dimensions must be positive integers")
        # Reuse unchanged algorithm checks via an explicit validation template.
        # This template is never passed to a trainer or saved in a checkpoint.
        from dataclasses import asdict
        if self.method != "B_PUBLISHED" or self.method_key != "B_PUBLISHED__b025":
            raise ValueError("Only the registered MATCH25% method is supported")
        from .recurrent_published_training_v5 import RecurrentMatchConfig
        base = RecurrentMatchConfig(phase="development", seed=0, method=self.method, **self.algorithm)
        base.validate()
        expected = set(asdict(base)) - {"phase", "seed", "method"}
        if set(self.algorithm) != expected:
            raise ValueError("Selection must resolve every algorithm field, including unchanged defaults")
        return self


def config_from_manifests(protocol_path, protocol_sha256, selection_path, selection_sha256, *, seed):
    protocol = read_hashed(protocol_path, protocol_sha256)
    selection = read_hashed(selection_path, selection_sha256)
    algorithm = dict(selection.get("hyperparameters", {}))
    steps = protocol.get("training_environment_steps")
    size = algorithm.get("num_envs", 0)*algorithm.get("horizon", 0)
    if type(steps) is not int or steps <= 0 or type(size) is not int or size <= 0 or steps % size:
        raise ValueError("Resolved shared step budget and integral rollout count are required")
    algorithm["num_updates"] = steps//size
    config = PublishedConfirmationConfig(protocol.get("phase"), seed, selection.get("method_key"), selection.get("method"),
        str(Path(protocol_path).resolve()), protocol_sha256, str(Path(selection_path).resolve()), selection_sha256, algorithm)
    config.validate_manifests()
    return config
