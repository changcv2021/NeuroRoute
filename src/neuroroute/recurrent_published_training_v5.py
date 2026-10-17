"""MATCH v5: full-batch actor replay before selecting a PPO minibatch.

The existing 128-environment loss, shuffle order and optimizer schedule remain.
All 512 actor histories are recomputed under current parameters at each
minibatch; only selected rows contribute to the loss. No old-parameter cache
or discarded action density is introduced. Extra forward computation keeps
actor array shapes consistent with the behavior policy.
"""
import hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
import optax
from .communication import FULL_BITS
from .published_policy import reconstruction_loss
from .published_training import canonical_arrays
from .training import generalized_advantage, ppo_objective
from .recurrent_published_training_v4 import (RecurrentMatchTrainer as V4Trainer,
    RecurrentMatchConfig, load_recurrent_config, implementation_fingerprint as v4_fingerprint)


def implementation_fingerprint():
    return hashlib.sha256(v4_fingerprint().encode()+Path(__file__).read_bytes()).hexdigest()


class RecurrentMatchTrainer(V4Trainer):
    def __init__(self, env, config, **kwargs):
        super().__init__(env, config, **kwargs)
        self.implementation_sha256 = implementation_fingerprint()
        self._compiled_selected_replay = jax.jit(self._selected_replay)

    def _checkpoint_contract(self):
        contract = super()._checkpoint_contract()
        contract.update(format="neuroroute.match.recursive_fullprefix.full_batch_forward.v5",
            actor_forward_batch=self.config.num_envs,
            minibatch_selection="after_complete_current_parameter_full_batch_replay")
        return contract

    def _selected_replay(self, params, prefix, lengths, rollout, key, ids):
        if rollout.global_obs.shape[1] != self.config.num_envs:
            raise ValueError("V5 minibatches must be selected after full registered actor replay")
        output = self._replay(params, prefix, lengths, rollout, key)
        output = jax.tree.map(jax.lax.optimization_barrier, output)
        return jax.tree.map(lambda x: x[:, ids], output)

    def _selected_loss(self, params, prefix, lengths, rollout, advantage, returns, dual, key, ids):
        policy, positions, attributes = self._selected_replay(params['actor'], prefix, lengths, rollout, key, ids)
        selected = jax.tree.map(lambda x: x[:, ids], rollout)
        prediction = self.critic.apply({'params': params['critic']}, selected.critic_obs)
        objective, metrics = ppo_objective(self._flatten(policy), self._flatten(prediction),
            self._flatten(selected), self._flatten(advantage[:, ids]), self._flatten(returns[:, ids]), dual, self.config)
        aux = reconstruction_loss(positions, attributes, selected.targets,
            position_scale=self.config.position_scale,
            attribute_scale=jnp.asarray(self.config.attribute_scale, jnp.float32))
        loss = objective+self.config.eta_reconstruction*aux['total'].mean()
        metrics.update(loss=loss, ppo_loss=objective, reconstruction_loss=aux['total'].mean(),
            reconstruction_position_loss=aux['position'].mean(), reconstruction_attribute_loss=aux['attribute'].mean())
        return loss, metrics

    def _update(self, state):
        prefix, lengths = state.prefix, state.episode_steps
        state, rollout = self._collect(state)
        advantage, returns = generalized_advantage(rollout.rewards, rollout.values, rollout.next_values,
            rollout.terminated, rollout.truncated, self.config.gamma, self.config.gae_lambda)

        def minibatch(carry, ids):
            params, optimizer, key = carry
            key, ek = jax.random.split(key)
            (_, metrics), gradient = jax.value_and_grad(self._selected_loss, has_aux=True)(params,
                prefix, lengths, rollout, advantage, returns, state.dual, ek, ids)
            changes, optimizer = self.optimizer.update(gradient, optimizer, params)
            params = optax.apply_updates(params, changes)
            metrics["gradient_norm"] = optax.global_norm(gradient)
            return (params, optimizer, key), metrics

        def epoch(carry, unused):
            params, optimizer, key = carry
            key, pk = jax.random.split(key)
            ids = jax.random.permutation(pk, self.config.num_envs).reshape((-1, self.config.minibatch_envs))
            return jax.lax.scan(minibatch, (params, optimizer, key), ids)

        (params, optimizer, key), metrics = jax.lax.scan(epoch,
            (state.params, state.optimizer_state, state.rng), None, length=self.config.epochs)
        metrics = jax.tree.map(jnp.mean, metrics)
        mean_cost = rollout.rewards[..., 1].mean()
        dual = jnp.maximum(jnp.array(0., jnp.float32), state.dual + self.config.dual_learning_rate *
                           (mean_cost - self.config.budget_fraction))
        # Recompute ongoing episode carry AFTER optimization before next rollout.
        refreshed = self._replay_prefix(params["actor"], state.prefix, state.episode_steps)
        size = self.config.num_envs * self.config.horizon
        costs = rollout.wire_bits[..., 0]
        metrics.update(mean_task_reward=rollout.rewards[..., 0].mean(), mean_bits_per_step=costs.mean(),
            total_bits=costs.sum(), up_bits=rollout.wire_bits[..., 1].sum(), down_bits=rollout.wire_bits[..., 2].sum(),
            bits_per_second=costs.sum() / (size * self.env.dt),
            budget_excess_bits=costs.mean() - FULL_BITS * self.config.budget_fraction,
            terminated_episodes=rollout.terminated.sum(), truncated_episodes=rollout.truncated.sum(), dual=dual,
            prefix_steps_max=state.episode_steps.max(), prefix_steps_mean=state.episode_steps.mean(),
            packet_gates=jnp.concatenate((rollout.action.up_gate, rollout.action.down_gate), axis=-1),
            raw_task_reward=rollout.rewards[..., 0], episode_terminated=rollout.terminated,
            episode_truncated=rollout.truncated)
        return canonical_arrays(state.replace(params=params, optimizer_state=optimizer, rng=key,
            match_state=refreshed, dual=dual, update_index=state.update_index + jnp.array(1, jnp.int32))), metrics

