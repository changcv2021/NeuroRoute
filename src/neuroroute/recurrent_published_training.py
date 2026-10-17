"""MATCH recursive packet/PPO: complete forward prefix replay, explicit TBPTT.

Every minibatch recomputes its reset-to-rollout prefix under CURRENT weights.
Only gradients at the rollout boundary are truncated; forward state and joint
likelihood are never approximated with an old-parameter prefix state.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from .communication import FULL_BITS, bit_cost
from .fast_reproducible_training import fast_numerical_runtime
from .policy import HybridAction
from .published_policy import reconstruction_loss
from .published_training import MatchTrainConfig, canonical_arrays, go1_target_extractor
from .recurrent_published_policy import (RecurrentPublishedPolicy, empty_recurrent_match_state,
                                          reset_recurrent_match_state)
from .training import ValueNetwork, generalized_advantage, ppo_objective, select_environments


@dataclass(frozen=True)
class RecurrentMatchConfig(MatchTrainConfig):
    minibatch_envs: int = 2
    prefix_gradient: str = "stop_at_rollout_boundary_full_current_forward"

    def validate(self):
        super().validate()
        if (self.minibatch_envs <= 0 or self.num_envs % self.minibatch_envs
                or self.minibatch_size != self.minibatch_envs * self.horizon):
            raise ValueError("Minibatches must contain complete per-environment rollout sequences")
        if self.prefix_gradient != "stop_at_rollout_boundary_full_current_forward":
            raise ValueError("Only explicit gradient truncation with exact current prefix replay is implemented")


@struct.dataclass
class Prefix:
    global_obs: jnp.ndarray
    local_obs: jnp.ndarray
    action: HybridAction
    up_received: jnp.ndarray
    down_received: jnp.ndarray


@struct.dataclass
class RecurrentTrainState:
    params: object
    optimizer_state: object
    env_state: object
    match_state: object
    prefix: Prefix
    rng: jnp.ndarray
    dual: jnp.ndarray
    update_index: jnp.ndarray
    environment_steps: jnp.ndarray
    episode_steps: jnp.ndarray


@struct.dataclass
class RecurrentTransition:
    global_obs: jnp.ndarray
    local_obs: jnp.ndarray
    action: HybridAction
    up_received: jnp.ndarray
    down_received: jnp.ndarray
    old_log_prob: jnp.ndarray
    critic_obs: jnp.ndarray
    targets: object
    values: jnp.ndarray
    next_values: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    wire_bits: jnp.ndarray


def empty_prefix(config):
    shape = (config.num_envs, config.episode_length)
    return Prefix(jnp.zeros(shape + (12,), jnp.float32), jnp.zeros(shape + (4, 9), jnp.float32),
        HybridAction(jnp.zeros(shape + (4,), jnp.bool_), jnp.zeros(shape + (4, 8), jnp.uint8),
                     jnp.zeros(shape + (4,), jnp.bool_), jnp.zeros(shape + (4, 8), jnp.uint8),
                     jnp.zeros(shape + (4, 3), jnp.float32)),
        jnp.zeros(shape + (4,), jnp.bool_), jnp.zeros(shape + (4,), jnp.bool_))


def implementation_fingerprint():
    files = ("recurrent_published_training.py", "recurrent_published_policy.py", "published_training.py",
             "published_policy.py", "training.py", "policy.py", "communication.py", "environment.py",
             "fast_reproducible_training.py", "reproducible_training.py")
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


class RecurrentMatchTrainer:
    def __init__(self, env, config, *, split_observation=None, assemble_action=None, target_extractor=None):
        config.validate()
        self.env, self.config = env, config
        self.runtime = fast_numerical_runtime()
        if split_observation is None or assemble_action is None:
            from .environment import assemble_actions, split_observations
            split_observation = lambda obs: split_observations(obs, env.leg_map)
            assemble_action = lambda act: assemble_actions(act, env.leg_map)
        self.split_observation, self.assemble_action = split_observation, assemble_action
        self.target_extractor = target_extractor or go1_target_extractor(env)
        self.actor = RecurrentPublishedPolicy(config.width, config.heads, config.layers)
        self.critic = ValueNetwork(config.critic_width)
        self.optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(config.learning_rate, eps=1e-5))
        self.implementation_sha256 = implementation_fingerprint()
        self.environment_signature = json.dumps({
            "class": type(env).__module__ + "." + type(env).__qualname__,
            "config": env._config.to_dict() if hasattr(env, "_config") else {}, "dt": float(env.dt),
            "fault": {name: getattr(env, name, None) for name in
                      ("fault_kind", "fault_leg", "retained", "onset_step", "freeze_steps")}}, sort_keys=True)
        self._compiled_initialize = jax.jit(self._initialize)
        self._compiled_update = jax.jit(self._update)
        self._compiled_collect = jax.jit(self._collect)
        self._compiled_prefix = jax.jit(self._replay_prefix)
        self._compiled_replay = jax.jit(self._replay)

    def _empty_history(self, count):
        return empty_recurrent_match_state((count,), self.config.history_length, self.config.width)

    def initialize(self):
        return canonical_arrays(self._compiled_initialize(jax.random.PRNGKey(self.config.seed)))

    def _initialize(self, key):
        key, ek, ak, vk, sk = jax.random.split(key, 5)
        env = canonical_arrays(jax.vmap(self.env.reset)(jax.random.split(ek, self.config.num_envs)))
        history = self._empty_history(self.config.num_envs)
        glob, local = self.split_observation(env.obs)
        actor = self.actor.init(ak, glob, local, history, sk, self.env.dt)["params"]
        critic = self.critic.init(vk, env.obs["privileged_state"])["params"]
        params = {"actor": actor, "critic": critic}
        return RecurrentTrainState(params, self.optimizer.init(params), env, history, empty_prefix(self.config),
            key, jnp.array(0., jnp.float32), jnp.array(0, jnp.int32), jnp.array(0, jnp.int32),
            jnp.zeros((self.config.num_envs,), jnp.int32))

    def _replay_prefix(self, actor_params, prefix, lengths):
        # Full current-parameter FORWARD reconstruction, no old learned carry.
        # Stopping parameter gradients here implements the declared TBPTT only.
        actor_params = jax.tree.map(jax.lax.stop_gradient, actor_params)
        empty = self._empty_history(lengths.shape[0])

        def step(index, state):
            item = jax.tree.map(lambda x: x[:, index], prefix)
            output = self.actor.apply({"params": actor_params}, item.global_obs, item.local_obs,
                state, jax.random.fold_in(jax.random.PRNGKey(713), index), self.env.dt,
                action=item.action, up_received=item.up_received, down_received=item.down_received)
            return select_environments(index < lengths, output.state, state)

        return jax.lax.fori_loop(0, jnp.max(lengths), step, empty)

    def _replay(self, actor_params, prefix, lengths, rollout, entropy_key):
        initial = self._replay_prefix(actor_params, prefix, lengths)

        def step(state, item):
            row, index = item
            output = self.actor.apply({"params": actor_params}, row.global_obs, row.local_obs, state,
                jax.random.fold_in(entropy_key, index), self.env.dt, action=row.action,
                up_received=row.up_received, down_received=row.down_received)
            done = row.terminated.astype(jnp.bool_) | row.truncated.astype(jnp.bool_)
            return reset_recurrent_match_state(output.state, done), (
                output.policy, output.positions, output.attributes)

        return jax.lax.scan(step, initial, (rollout, jnp.arange(self.config.horizon)))[1]

    def _rollout_step(self, state, unused):
        key, action_key, reset_key = jax.random.split(state.rng, 3)
        glob, local = self.split_observation(state.env_state.obs)
        links = jnp.ones((self.config.num_envs, 4), jnp.bool_)
        output = self.actor.apply({"params": state.params["actor"]}, glob, local, state.match_state,
            action_key, self.env.dt, up_received=links, down_received=links)
        critic_obs = state.env_state.obs["privileged_state"]
        values = self.critic.apply({"params": state.params["critic"]}, critic_obs)
        targets = self.target_extractor(state.env_state)
        next_env = canonical_arrays(jax.vmap(self.env.step)(state.env_state,
            self.assemble_action(output.policy.motor_action)))
        next_values = self.critic.apply({"params": state.params["critic"]}, next_env.obs["privileged_state"])
        steps = state.episode_steps + jnp.array(1, jnp.int32)
        terminated = next_env.done.astype(jnp.bool_)
        truncated = (steps >= self.config.episode_length) & ~terminated
        done = terminated | truncated
        cost = bit_cost(output.policy.action.up_gate, output.policy.action.down_gate)
        row = RecurrentTransition(glob, local, output.policy.action, links, links, output.policy.log_prob,
            critic_obs, targets, values, next_values, jnp.stack((next_env.reward, cost.total / FULL_BITS), axis=-1),
            terminated.astype(jnp.float32), truncated.astype(jnp.float32),
            jnp.stack((cost.total, cost.up, cost.down), axis=-1))
        recorded = Prefix(glob, local, output.policy.action, links, links)
        prefix = jax.tree.map(lambda old, value: old.at[jnp.arange(self.config.num_envs), state.episode_steps].set(value),
                              state.prefix, recorded)
        prefix = jax.tree.map(lambda x: jnp.where(done.reshape(done.shape + (1,) * (x.ndim - 1)),
                                                 jnp.zeros_like(x), x), prefix)

        def reset_env(current):
            fresh = canonical_arrays(jax.vmap(self.env.reset)(jax.random.split(reset_key, self.config.num_envs)))
            return select_environments(done, fresh, current)

        next_env = jax.lax.cond(jnp.any(done), reset_env, lambda x: x, next_env)
        return state.replace(env_state=next_env, match_state=reset_recurrent_match_state(output.state, done),
            prefix=prefix, rng=key, episode_steps=jnp.where(done, jnp.array(0, jnp.int32), steps),
            environment_steps=state.environment_steps + jnp.array(self.config.num_envs, jnp.int32)), row

    def _collect(self, state):
        return jax.lax.scan(self._rollout_step, state, None, length=self.config.horizon)

    @staticmethod
    def _flatten(tree):
        return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), tree)

    def _loss(self, params, prefix, lengths, rollout, advantage, returns, dual, key):
        policy, positions, attributes = self._replay(params["actor"], prefix, lengths, rollout, key)
        prediction = self.critic.apply({"params": params["critic"]}, rollout.critic_obs)
        objective, metrics = ppo_objective(self._flatten(policy), self._flatten(prediction),
            self._flatten(rollout), self._flatten(advantage), self._flatten(returns), dual, self.config)
        aux = reconstruction_loss(positions, attributes, rollout.targets,
            position_scale=self.config.position_scale,
            attribute_scale=jnp.asarray(self.config.attribute_scale, jnp.float32))
        loss = objective + self.config.eta_reconstruction * aux["total"].mean()
        metrics.update(loss=loss, ppo_loss=objective, reconstruction_loss=aux["total"].mean(),
            reconstruction_position_loss=aux["position"].mean(), reconstruction_attribute_loss=aux["attribute"].mean())
        return loss, metrics

    def _update(self, state):
        prefix, lengths = state.prefix, state.episode_steps
        state, rollout = self._collect(state)
        advantage, returns = generalized_advantage(rollout.rewards, rollout.values, rollout.next_values,
            rollout.terminated, rollout.truncated, self.config.gamma, self.config.gae_lambda)

        def minibatch(carry, ids):
            params, optimizer, key = carry
            key, ek = jax.random.split(key)
            selected_prefix = jax.tree.map(lambda x: x[ids], prefix)
            selected_rollout = jax.tree.map(lambda x: x[:, ids], rollout)
            (_, metrics), gradient = jax.value_and_grad(self._loss, has_aux=True)(params,
                selected_prefix, lengths[ids], selected_rollout, advantage[:, ids], returns[:, ids], state.dual, ek)
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

    def update(self, state):
        state, metrics = self._compiled_update(canonical_arrays(state))
        metrics = jax.device_get(metrics)
        if not all(np.all(np.isfinite(x)) for x in metrics.values()):
            raise FloatingPointError("Nonfinite recurrent MATCH update")
        return canonical_arrays(state), metrics

    def verify_rollout_replay(self, state):
        _, rollout = self._compiled_collect(canonical_arrays(state))
        policy, _, _ = self._compiled_replay(state.params["actor"], state.prefix, state.episode_steps,
                                            rollout, jax.random.PRNGKey(618))
        return float(jnp.max(jnp.abs(policy.log_prob - rollout.old_log_prob))), rollout

    def save_checkpoint(self, path, state):
        if self.runtime != fast_numerical_runtime():
            raise ValueError("Runtime changed after initialization")
        if any(bool(getattr(x, "weak_type", False)) for x in jax.tree.leaves(state)):
            raise ValueError("Weakly typed recurrent state cannot be saved")
        payload = {"format": "neuroroute.match.recursive_fullprefix.v1", "config": asdict(self.config),
            "runtime": self.runtime, "environment_signature": self.environment_signature,
            "implementation_sha256": self.implementation_sha256, "state": jax.device_get(state),
            "prefix_forward": "complete_current_parameters_from_reset", "prefix_gradient": self.config.prefix_gradient}
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
        expected = {"format": "neuroroute.match.recursive_fullprefix.v1", "config": asdict(self.config),
            "runtime": self.runtime, "environment_signature": self.environment_signature,
            "implementation_sha256": self.implementation_sha256,
            "prefix_forward": "complete_current_parameters_from_reset", "prefix_gradient": self.config.prefix_gradient}
        if self.runtime != fast_numerical_runtime() or any(payload.get(k) != v for k, v in expected.items()):
            raise ValueError("Recursive MATCH checkpoint/runtime mismatch; legacy formats are not migrated")
        state = canonical_arrays(payload["state"])
        lengths = np.asarray(state.episode_steps)
        if np.any(lengths < 0) or np.any(lengths >= self.config.episode_length):
            raise ValueError("Invalid reset-to-current prefix length")
        return state


def load_recurrent_config(path):
    values = json.loads(Path(path).read_text())
    for name in ("attribute_scale", "pi_bio"):
        if name in values:
            values[name] = tuple(values[name])
    config = RecurrentMatchConfig(**values)
    config.validate()
    return config
