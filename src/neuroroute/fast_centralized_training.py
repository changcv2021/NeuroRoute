"""Central PPO on the separately validated JAX082 numerical runtime."""
import hashlib
import json
from pathlib import Path

import jax
import optax

from .centralized_training import CentralPPOTrainer, CentralActor, CentralCritic, source_fingerprint
from .fast_reproducible_training import fast_numerical_runtime


class FastCentralPPOTrainer(CentralPPOTrainer):
    def __init__(self, env, config, *, assemble_action=None):
        config.validate()
        self.env, self.config = env, config
        self.runtime = fast_numerical_runtime()
        source = (source_fingerprint().encode() + Path(__file__).read_bytes()
                  + Path(__file__).with_name("fast_reproducible_training.py").read_bytes())
        self.implementation_sha256 = hashlib.sha256(
            source + json.dumps(self.runtime, sort_keys=True).encode()).hexdigest()
        if assemble_action is None:
            from .environment import assemble_actions
            assemble_action = lambda action: assemble_actions(action, env.leg_map)
        self.assemble_action = assemble_action
        self.actor = CentralActor(config.actor_width)
        self.critic = CentralCritic(config.critic_width)
        self.optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(config.learning_rate, eps=1e-5))
        self._compiled_initialize = jax.jit(self._initialize)
        self._compiled_update = jax.jit(self._update)
