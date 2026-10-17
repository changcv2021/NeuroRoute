"""Development joint PPO for the explicit MATCH packet adaptation.

Every transition stores the complete pre-action raw MatchState. Shuffling a
minibatch never reconstructs history by repeating the current packet or using
stale learned hidden states. This is not the original MATCH Q/TD algorithm;
original-task replication and confirmation experiments remain outstanding.
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

from .checkpoint_runtime import checkpoint_runtime
from .communication import FULL_BITS, bit_cost
from .published_policy import (PublishedMatchPolicy, empty_match_state,
                               privileged_targets, reconstruction_loss,
                               reset_match_state)
from .training import (ValueNetwork, generalized_advantage, ppo_objective,
                       select_environments)


@dataclass(frozen=True)
class MatchTrainConfig:
    phase: str = "development"
    method: str = "B_PUBLISHED"
    seed: int = 0
    num_envs: int = 4
    horizon: int = 8
    num_updates: int = 2
    epochs: int = 1
    minibatch_size: int = 16
    history_length: int = 3
    episode_length: int = 1000
    width: int = 32
    heads: int = 4
    layers: int = 1
    critic_width: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_gate: float = 1e-3
    entropy_message: float = 1e-4
    entropy_motor: float = 1e-3
    max_grad_norm: float = 1.0
    budget_fraction: float = 0.25
    dual_learning_rate: float = 0.05
    eta_reconstruction: float = 0.1
    position_scale: float = 0.3
    attribute_scale: tuple = (1.0, 1.0, 1.0, 10.0, 10.0, 10.0)
    normalization: str = "fixed_identity"
    # Required by the tested shared PPO objective; explicitly never biological.
    lambda_bio: float = 0.0
    pi_bio: tuple = (0.5, 0.5)

    def validate(self):
        if self.phase != "development" or self.method != "B_PUBLISHED" or self.seed not in (0, 1):
            raise ValueError("MATCH adapter supports development B_PUBLISHED seeds 0/1 only")
        for field in ("num_envs", "horizon", "num_updates", "epochs", "minibatch_size",
                      "history_length", "episode_length", "width", "heads", "layers", "critic_width"):
            if not isinstance(getattr(self, field), int) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.num_envs * self.horizon % self.minibatch_size:
            raise ValueError("Rollout size must divide into complete minibatches")
        if self.width % self.heads:
            raise ValueError("Transformer width must be divisible by head count")
        if self.budget_fraction not in (0.25, 0.5):
            raise ValueError("MATCH development uses a 25% or 50% average wire budget")
        if self.lambda_bio != 0 or tuple(self.pi_bio) != (0.5, 0.5):
            raise ValueError("Published baseline cannot receive a biological prior")
        if self.normalization != "fixed_identity":
            raise ValueError("Actor input normalization must remain fixed identity")
        if (self.eta_reconstruction <= 0 or self.position_scale <= 0
                or len(self.attribute_scale) != 6 or min(self.attribute_scale) <= 0):
            raise ValueError("Reconstruction loss and six field scales must be positive")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("Invalid GAE configuration")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or not 0 < self.clip_epsilon < 1:
            raise ValueError("Invalid optimizer configuration")


@struct.dataclass
class MatchTrainState:
    params: object
    optimizer_state: object
    env_state: object
    match_state: object
    rng: jnp.ndarray
    dual: jnp.ndarray
    update_index: jnp.ndarray
    environment_steps: jnp.ndarray
    episode_steps: jnp.ndarray


@struct.dataclass
class MatchTransition:
    global_obs: jnp.ndarray
    local_obs: jnp.ndarray
    match_state: object
    action: object
    old_log_prob: jnp.ndarray
    critic_obs: jnp.ndarray
    targets: object
    values: jnp.ndarray
    next_values: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    wire_bits: jnp.ndarray


def implementation_fingerprint():
    digest = hashlib.sha256()
    for name in ("published_training.py", "published_policy.py", "training.py",
                 "policy.py", "communication.py", "environment.py",
                 "checkpoint_runtime.py", "reproducible_training.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def canonical_arrays(tree):
    """All leaves have explicit dtypes across init, checkpoint and JIT calls."""
    return jax.tree.map(lambda x: jax.lax.convert_element_type(x, jnp.asarray(x).dtype), tree)


def match_critic_input(observation, match_state):
    """Privileged critic only; no result from here is an actor input."""
    privileged = observation["privileged_state"]
    batch = privileged.shape[:-1]
    arrays = [privileged]
    for value in jax.tree.leaves(match_state):
        decoded = value.astype(jnp.float32)
        if value.dtype == jnp.uint8:
            decoded = decoded / 255.0
        arrays.append(decoded.reshape(batch + (-1,)))
    return jnp.concatenate(arrays, axis=-1)


def go1_target_extractor(env):
    """Discover foot sites and body geometry once; labels are used only in loss."""
    from .environment import LEGS
    foot_ids = np.array([env.mj_model.site(name).id for name in LEGS])
    torso_id = env.mj_model.body("trunk").id
    mapping = env.leg_map

    def targets(state):
        data = state.data
        q = data.qpos[..., 7 + mapping.qpos_local]
        dq = data.qvel[..., 6 + mapping.qvel_local]
        feet_world = data.site_xpos[..., foot_ids, :]
        torso_world = data.xpos[..., torso_id, :]
        torso_rotation = data.xmat[..., torso_id, :, :]
        # R maps local to world; row-vector multiplication by R maps back.
        feet_local = jnp.einsum("...ij,...jk->...ik", feet_world - torso_world[..., None, :], torso_rotation)
        return privileged_targets(q, dq, feet_local)

    return targets


class MatchPPOTrainer:
    def __init__(self, env, config, *, split_observation=None, assemble_action=None,
                 target_extractor=None):
        config.validate()
        self.env, self.config = env, config
        if split_observation is None or assemble_action is None:
            from .environment import assemble_actions, split_observations
            split_observation = lambda obs: split_observations(obs, env.leg_map)
            assemble_action = lambda act: assemble_actions(act, env.leg_map)
        self.split_observation, self.assemble_action = split_observation, assemble_action
        self.target_extractor = target_extractor or go1_target_extractor(env)
        self.actor = PublishedMatchPolicy(config.width, config.heads, config.layers)
        self.critic = ValueNetwork(config.critic_width)
        self.optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(config.learning_rate, eps=1e-5))
        self.implementation_sha256 = implementation_fingerprint()
        self.runtime = checkpoint_runtime()
        env_config = env._config.to_dict() if hasattr(env, "_config") else {}
        self.environment_signature = json.dumps({
            "class": type(env).__module__ + "." + type(env).__qualname__,
            "config": env_config, "dt": float(env.dt),
            "fault": {name: getattr(env, name, None) for name in
                      ("fault_kind", "fault_leg", "retained", "onset_step", "freeze_steps")}}, sort_keys=True)
        self._compiled_initialize = jax.jit(self._initialize)
        self._compiled_update = jax.jit(self._update)
        self._compiled_collect = jax.jit(self._collect)
        self._compiled_replay_error = jax.jit(self._replay_error)

    def initialize(self):
        return canonical_arrays(self._compiled_initialize(jax.random.PRNGKey(self.config.seed)))

    def _initialize(self, key):
        key, ek, ak, sk, vk = jax.random.split(key, 5)
        env = canonical_arrays(jax.vmap(self.env.reset)(jax.random.split(ek, self.config.num_envs)))
        global_obs, local_obs = self.split_observation(env.obs)
        history = empty_match_state((self.config.num_envs,), self.config.history_length)
        actor = self.actor.init(ak, global_obs, local_obs, history, sk, self.env.dt)["params"]
        critic = self.critic.init(vk, match_critic_input(env.obs, history))["params"]
        params = {"actor": actor, "critic": critic}
        return MatchTrainState(params, self.optimizer.init(params), env, history, key,
                               jnp.array(0.0, jnp.float32), jnp.array(0, jnp.int32),
                               jnp.array(0, jnp.int32), jnp.zeros((self.config.num_envs,), jnp.int32))

    def _rollout_step(self, state, unused):
        key, sample_key, reset_key = jax.random.split(state.rng, 3)
        global_obs, local_obs = self.split_observation(state.env_state.obs)
        output = self.actor.apply({"params": state.params["actor"]}, global_obs, local_obs,
                                  state.match_state, sample_key, self.env.dt)
        ci = match_critic_input(state.env_state.obs, state.match_state)
        values = self.critic.apply({"params": state.params["critic"]}, ci)
        # Critic/reconstruction labels are extracted independently after the
        # actor invocation; there is no simulator truth in its call signature.
        targets = self.target_extractor(state.env_state)
        next_env = canonical_arrays(jax.vmap(self.env.step)(state.env_state, self.assemble_action(output.policy.motor_action)))
        next_ci = match_critic_input(next_env.obs, output.state)
        next_values = self.critic.apply({"params": state.params["critic"]}, next_ci)
        steps = state.episode_steps + jnp.array(1, jnp.int32)
        terminated = next_env.done.astype(jnp.bool_)
        truncated = (steps >= self.config.episode_length) & ~terminated
        done = terminated | truncated
        cost = bit_cost(output.policy.action.up_gate, output.policy.action.down_gate)
        rewards = jnp.stack((next_env.reward, cost.total / FULL_BITS), axis=-1)
        transition = MatchTransition(global_obs, local_obs, state.match_state,
            output.policy.action, output.policy.log_prob, ci, targets, values, next_values,
            rewards, terminated.astype(jnp.float32), truncated.astype(jnp.float32),
            jnp.stack((cost.total, cost.up, cost.down), axis=-1))

        def reset_ended(current):
            fresh = canonical_arrays(jax.vmap(self.env.reset)(jax.random.split(reset_key, self.config.num_envs)))
            return select_environments(done, fresh, current)

        next_env = jax.lax.cond(jnp.any(done), reset_ended, lambda x: x, next_env)
        history = reset_match_state(output.state, done)
        return state.replace(env_state=next_env, match_state=history, rng=key,
            episode_steps=jnp.where(done, jnp.array(0, jnp.int32), steps),
            environment_steps=state.environment_steps + jnp.array(self.config.num_envs, jnp.int32)), transition

    def _collect(self, state):
        return jax.lax.scan(self._rollout_step, state, None, length=self.config.horizon)

    @staticmethod
    def _flatten(rollout):
        return jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), rollout)

    def _loss(self, params, selected, advantages, returns, dual, entropy_key):
        output = self.actor.apply({"params": params["actor"]}, selected.global_obs,
            selected.local_obs, selected.match_state, entropy_key, self.env.dt,
            action=selected.action)
        predicted = self.critic.apply({"params": params["critic"]}, selected.critic_obs)
        objective, metrics = ppo_objective(output.policy, predicted, selected, advantages,
                                           returns, dual, self.config)
        reconstruction = reconstruction_loss(output.positions, output.attributes, selected.targets,
            position_scale=self.config.position_scale,
            attribute_scale=jnp.asarray(self.config.attribute_scale, jnp.float32))
        auxiliary = reconstruction["total"].mean()
        total = objective + self.config.eta_reconstruction * auxiliary
        metrics.update(loss=total, ppo_loss=objective, reconstruction_loss=auxiliary,
                       reconstruction_position_loss=reconstruction["position"].mean(),
                       reconstruction_attribute_loss=reconstruction["attribute"].mean())
        return total, metrics

    def _update(self, state):
        state, rollout = self._collect(state)
        advantage, returns = generalized_advantage(rollout.rewards, rollout.values,
            rollout.next_values, rollout.terminated, rollout.truncated,
            self.config.gamma, self.config.gae_lambda)
        batch, advantage, returns = self._flatten((rollout, advantage, returns))
        size = self.config.num_envs * self.config.horizon

        def minibatch(carry, ids):
            params, optimizer, key = carry
            key, entropy_key = jax.random.split(key)
            selected = jax.tree.map(lambda x: x[ids], batch)
            (_, metrics), grad = jax.value_and_grad(self._loss, has_aux=True)(
                params, selected, advantage[ids], returns[ids], state.dual, entropy_key)
            updates, optimizer = self.optimizer.update(grad, optimizer, params)
            params = optax.apply_updates(params, updates)
            metrics["gradient_norm"] = optax.global_norm(grad)
            return (params, optimizer, key), metrics

        def epoch(carry, unused):
            params, optimizer, key = carry
            key, permutation = jax.random.split(key)
            indices = jax.random.permutation(permutation, size).reshape((-1, self.config.minibatch_size))
            return jax.lax.scan(minibatch, (params, optimizer, key), indices)

        (params, optimizer, key), raw_metrics = jax.lax.scan(epoch,
            (state.params, state.optimizer_state, state.rng), None, length=self.config.epochs)
        metrics = jax.tree.map(jnp.mean, raw_metrics)
        mean_cost = rollout.rewards[..., 1].mean()
        dual = jnp.maximum(jnp.array(0.0, jnp.float32), state.dual + self.config.dual_learning_rate * (mean_cost - self.config.budget_fraction))
        costs = rollout.wire_bits[..., 0]
        metrics.update(mean_task_reward=rollout.rewards[..., 0].mean(),
            mean_bits_per_step=costs.mean(), total_bits=costs.sum(),
            up_bits=rollout.wire_bits[..., 1].sum(), down_bits=rollout.wire_bits[..., 2].sum(),
            bits_per_second=costs.sum() / (size * self.env.dt),
            budget_excess_bits=costs.mean() - FULL_BITS * self.config.budget_fraction,
            terminated_episodes=rollout.terminated.sum(), truncated_episodes=rollout.truncated.sum(),
            dual=dual, packet_gates=jnp.concatenate((rollout.action.up_gate, rollout.action.down_gate), axis=-1),
            raw_task_reward=rollout.rewards[..., 0], episode_terminated=rollout.terminated,
            episode_truncated=rollout.truncated)
        return state.replace(params=params, optimizer_state=optimizer, rng=key, dual=dual,
            update_index=state.update_index + jnp.array(1, jnp.int32)), metrics

    def update(self, state):
        state, metrics = self._compiled_update(canonical_arrays(state))
        metrics = jax.device_get(metrics)
        if not all(np.all(np.isfinite(value)) for value in metrics.values()):
            raise FloatingPointError("Nonfinite MATCH development update; preserve prior checkpoint")
        return canonical_arrays(state), metrics

    def _replay_error(self, params, rollout):
        batch = self._flatten(rollout)
        output = self.actor.apply({"params": params["actor"]}, batch.global_obs,
            batch.local_obs, batch.match_state, jax.random.PRNGKey(931), self.env.dt,
            action=batch.action)
        return jnp.max(jnp.abs(output.policy.log_prob - batch.old_log_prob))

    def verify_rollout_replay(self, state):
        """Diagnostic rollout from a fork; original state and optimizer unchanged."""
        _, rollout = self._compiled_collect(canonical_arrays(state))
        return float(self._compiled_replay_error(state.params, rollout)), rollout

    def save_checkpoint(self, path, state):
        """Trusted local pickle, including raw history and every optimizer slot."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = canonical_arrays(state)
        if checkpoint_runtime() != self.runtime:
            raise ValueError("MATCH numerical runtime changed after trainer initialization")
        payload = {"format": "neuroroute.match.development.v2", "config": asdict(self.config),
            "numerical_runtime": self.runtime,
            "implementation_sha256": self.implementation_sha256,
            "environment_signature": self.environment_signature,
            "state_leaf_weak_types": [bool(getattr(x, "weak_type", False)) for x in jax.tree.leaves(state)],
            "state": jax.device_get(state), "normalizer": "fixed_identity",
            "scheduler": {"kind": "constant", "learning_rate": self.config.learning_rate},
            "original_task_reproduction_complete": False}
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
        """Only load files created by this trusted local runner."""
        with Path(path).open("rb") as stream:
            content = pickle.load(stream)
        if (content.get("format") != "neuroroute.match.development.v2"
                or content.get("numerical_runtime") != self.runtime
                or checkpoint_runtime() != self.runtime
                or content["config"] != asdict(self.config)
                or content["implementation_sha256"] != self.implementation_sha256
                or content["environment_signature"] != self.environment_signature
                or any(content["state_leaf_weak_types"])):
            raise ValueError("MATCH checkpoint/runtime/config/environment/dtype mismatch; v1 is not migrated")
        return canonical_arrays(content["state"])


def load_match_config(path):
    values = json.loads(Path(path).read_text())
    for name in ("attribute_scale", "pi_bio"):
        if name in values:
            values[name] = tuple(values[name])
    config = MatchTrainConfig(**values)
    config.validate()
    return config
