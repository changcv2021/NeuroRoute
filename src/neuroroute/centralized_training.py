"""Centralized development PPO reference with no packet policy or bandwidth point.

All 48 legal actor signals and three finite history frames enter the actor.
Privileged signals enter only the value network. Checkpoints are trusted local
pickle files; loading an untrusted pickle can execute arbitrary code.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import struct

from .policy import squashed_gaussian_log_prob
from .training import generalized_advantage, select_environments


def strong_types(tree):
    """Pinned JAX public conversion clears weak_type without changing values."""
    return jax.tree.map(lambda a: jax.lax.convert_element_type(a, a.dtype), tree)


def leaf_signatures(tree):
    return [{"path": jax.tree_util.keystr(path), "shape": list(a.shape),
             "dtype": str(a.dtype), "weak_type": bool(jax.core.get_aval(a).weak_type)}
            for path, a in jax.tree_util.tree_flatten_with_path(tree)[0]]


def source_fingerprint():
    digest = hashlib.sha256()
    for name in ("centralized_training.py", "training.py", "policy.py", "communication.py",
                 "environment.py", "reproducible_training.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CentralConfig:
    phase: str = "development"
    method: str = "B_CENTRAL"
    seed: int = 0
    num_envs: int = 16
    horizon: int = 32
    num_updates: int = 4
    epochs: int = 2
    minibatch_size: int = 128
    history_length: int = 3
    episode_length: int = 1000
    actor_width: int = 128
    critic_width: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_motor: float = 1e-3
    max_grad_norm: float = 1.0
    normalization: str = "fixed_identity"

    def validate(self):
        if self.phase != "development" or self.seed not in (0, 1) or self.method != "B_CENTRAL":
            raise ValueError("Only development B_CENTRAL seeds 0/1 are authorized by this implementation")
        for name in ("num_envs", "horizon", "num_updates", "epochs", "minibatch_size", "episode_length",
                     "actor_width", "critic_width"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_envs * self.horizon % self.minibatch_size:
            raise ValueError("Rollout size must be divisible by minibatch size")
        if self.history_length != 3 or self.normalization != "fixed_identity":
            raise ValueError("Central development locks 3-frame history and fixed identity normalization")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid GAE configuration")
        for name in ("learning_rate", "clip_epsilon", "value_coefficient", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.entropy_motor) or self.entropy_motor < 0:
            raise ValueError("Entropy coefficient must be nonnegative and finite")


@struct.dataclass
class CentralOutput:
    motor_raw: jnp.ndarray
    motor_action: jnp.ndarray
    motor_mean: jnp.ndarray
    log_prob: jnp.ndarray
    entropy: jnp.ndarray


class CentralActor(nn.Module):
    width: int = 128

    @nn.compact
    def __call__(self, actor_history, key, *, motor_raw=None, deterministic_motor=False):
        if actor_history.shape[-2:] != (3, 48):
            raise ValueError("Actor requires exactly three frames of the 48 legal signals")
        batch = actor_history.shape[:-2]
        x = actor_history.reshape(batch + (144,))
        x = nn.tanh(nn.Dense(self.width, name="hidden_0")(x))
        x = nn.tanh(nn.Dense(self.width, name="hidden_1")(x))
        mean = nn.Dense(12, kernel_init=nn.initializers.orthogonal(0.01), name="motor_mean")(x)
        mean = mean.reshape(batch + (4, 3))
        # Same three joint-role standard deviations shared across four legs
        # as in HybridPolicy, and the same downstream action transform.
        log_std = jnp.clip(self.param("motor_log_std", nn.initializers.zeros, (3,)), -5.0, 2.0)
        sample_key, entropy_key = jax.random.split(key)
        if motor_raw is None:
            motor_raw = mean if deterministic_motor else mean + jnp.exp(log_std) * jax.random.normal(sample_key, mean.shape)
        log_prob = squashed_gaussian_log_prob(motor_raw, mean, log_std)
        entropy_sample = mean + jnp.exp(log_std) * jax.random.normal(entropy_key, mean.shape)
        log_jacobian = 2.0 * (math.log(2.0) - entropy_sample - jax.nn.softplus(-2.0 * entropy_sample))
        entropy = 4.0 * jnp.sum(log_std + 0.5 * math.log(2.0 * math.pi * math.e)) + log_jacobian.sum(axis=(-1, -2))
        return CentralOutput(motor_raw, jnp.tanh(motor_raw), mean, log_prob, entropy)


class CentralCritic(nn.Module):
    width: int = 128

    @nn.compact
    def __call__(self, inputs):
        x = nn.tanh(nn.Dense(self.width)(inputs))
        x = nn.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)[..., 0]


@struct.dataclass
class CentralState:
    params: object
    optimizer_state: object
    env_state: object
    actor_history: jnp.ndarray
    rng: jnp.ndarray
    update_index: jnp.ndarray
    environment_steps: jnp.ndarray
    episode_steps: jnp.ndarray


@struct.dataclass
class CentralTransition:
    actor_history: jnp.ndarray
    motor_raw: jnp.ndarray
    old_log_prob: jnp.ndarray
    critic_obs: jnp.ndarray
    values: jnp.ndarray
    next_values: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray


def new_history(observation):
    actor = observation["state"]
    if actor.shape[-1] != 48:
        raise ValueError("Unaudited actor observation dimension")
    return jnp.repeat(actor[..., None, :], 3, axis=-2)


def advance_history(history, observation):
    return jnp.concatenate((history[..., 1:, :], observation["state"][..., None, :]), axis=-2)


def value_input(observation, history):
    return jnp.concatenate((observation["privileged_state"], history.reshape(history.shape[:-2] + (144,))), axis=-1)


class CentralPPOTrainer:
    def __init__(self, env, config: CentralConfig, *, assemble_action=None):
        config.validate()
        self.env, self.config = env, config
        from .reproducible_training import numerical_runtime
        self.runtime = numerical_runtime()
        self.implementation_sha256 = hashlib.sha256(
            source_fingerprint().encode() + json.dumps(self.runtime, sort_keys=True).encode()).hexdigest()
        if assemble_action is None:
            from .environment import assemble_actions
            assemble_action = lambda a: assemble_actions(a, env.leg_map)
        self.assemble_action = assemble_action
        self.actor = CentralActor(config.actor_width)
        self.critic = CentralCritic(config.critic_width)
        self.optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(config.learning_rate, eps=1e-5))
        self._compiled_initialize = jax.jit(self._initialize)
        self._compiled_update = jax.jit(self._update)

    def initialize(self):
        return self._compiled_initialize(jax.random.PRNGKey(self.config.seed))

    def _initialize(self, key):
        key, reset_key, actor_key, sample_key, critic_key = jax.random.split(key, 5)
        env_state = strong_types(jax.vmap(self.env.reset)(jax.random.split(reset_key, self.config.num_envs)))
        history = new_history(env_state.obs)
        params = {"actor": self.actor.init(actor_key, history, sample_key)["params"],
                  "critic": self.critic.init(critic_key, value_input(env_state.obs, history))["params"]}
        return strong_types(CentralState(params, self.optimizer.init(params), env_state, history, key,
            jnp.array(0, jnp.int32), jnp.array(0, jnp.int32), jnp.zeros(self.config.num_envs, jnp.int32)))

    def _rollout_step(self, state, unused):
        key, actor_key, reset_key = jax.random.split(state.rng, 3)
        output = self.actor.apply({"params": state.params["actor"]}, state.actor_history, actor_key)
        critic_obs = value_input(state.env_state.obs, state.actor_history)
        values = self.critic.apply({"params": state.params["critic"]}, critic_obs)
        after = jax.vmap(self.env.step)(state.env_state, self.assemble_action(output.motor_action))
        history = advance_history(state.actor_history, after.obs)
        next_values = self.critic.apply({"params": state.params["critic"]}, value_input(after.obs, history))
        steps = state.episode_steps + jnp.array(1, jnp.int32)
        terminated = after.done.astype(jnp.bool_)
        truncated = (steps >= self.config.episode_length) & ~terminated
        done = terminated | truncated
        transition = CentralTransition(state.actor_history, output.motor_raw, output.log_prob, critic_obs,
                                       values, next_values, after.reward, terminated.astype(jnp.float32),
                                       truncated.astype(jnp.float32))

        def reset_ended(args):
            after, history = args
            fresh = jax.vmap(self.env.reset)(jax.random.split(reset_key, self.config.num_envs))
            return select_environments(done, fresh, after), select_environments(done, new_history(fresh.obs), history)

        after, history = jax.lax.cond(jnp.any(done), reset_ended, lambda args: args, (after, history))
        next_state = state.replace(env_state=after, actor_history=history, rng=key,
            episode_steps=jnp.where(done, jnp.array(0, jnp.int32), steps),
            environment_steps=state.environment_steps + jnp.array(self.config.num_envs, jnp.int32))
        return strong_types(next_state), strong_types(transition)

    def _update(self, state):
        state, rollout = jax.lax.scan(self._rollout_step, state, None, length=self.config.horizon)
        advantages, returns = generalized_advantage(rollout.rewards[..., None], rollout.values[..., None],
            rollout.next_values[..., None], rollout.terminated, rollout.truncated,
            self.config.gamma, self.config.gae_lambda)
        flatten = lambda a: a.reshape((-1,) + a.shape[2:])
        batch = jax.tree.map(flatten, rollout)
        advantages, returns = advantages.reshape(-1), returns.reshape(-1)
        size = self.config.num_envs * self.config.horizon

        def minibatch(carry, indices):
            params, optimizer_state, key = carry
            key, entropy_key = jax.random.split(key)
            selected = jax.tree.map(lambda a: a[indices], batch)
            advantage = advantages[indices]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            def loss(params):
                output = self.actor.apply({"params": params["actor"]}, selected.actor_history,
                                           entropy_key, motor_raw=selected.motor_raw)
                predicted = self.critic.apply({"params": params["critic"]}, selected.critic_obs)
                log_ratio = output.log_prob - selected.old_log_prob
                ratio = jnp.exp(log_ratio)
                clipped = jnp.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                actor_loss = -jnp.minimum(ratio * advantage, clipped * advantage).mean()
                value_loss = 0.5 * jnp.square(predicted - returns[indices]).mean()
                total = actor_loss + self.config.value_coefficient * value_loss - self.config.entropy_motor * output.entropy.mean()
                return total, {"loss": total, "actor_loss": actor_loss, "task_value_loss": value_loss,
                               "entropy_motor": output.entropy.mean(), "approx_kl": ((ratio - 1) - log_ratio).mean(),
                               "clip_fraction": (jnp.abs(ratio - 1) > self.config.clip_epsilon).mean()}

            (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(params)
            updates, optimizer_state = self.optimizer.update(gradients, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            metrics["gradient_norm"] = optax.global_norm(gradients)
            return strong_types((params, optimizer_state, key)), strong_types(metrics)

        def epoch(carry, unused):
            params, optimizer_state, key = carry
            key, permutation_key = jax.random.split(key)
            indices = jax.random.permutation(permutation_key, size).reshape((-1, self.config.minibatch_size))
            return jax.lax.scan(minibatch, (params, optimizer_state, key), indices)

        (params, optimizer_state, key), metrics = jax.lax.scan(epoch,
            (state.params, state.optimizer_state, state.rng), None, length=self.config.epochs)
        metrics = jax.tree.map(jnp.mean, metrics)
        metrics.update(mean_task_reward=rollout.rewards.mean(),
            terminated_episodes=rollout.terminated.sum(), truncated_episodes=rollout.truncated.sum(),
            raw_task_reward=rollout.rewards, episode_terminated=rollout.terminated,
            episode_truncated=rollout.truncated)
        state = state.replace(params=params, optimizer_state=optimizer_state, rng=key,
                              update_index=state.update_index + jnp.array(1, jnp.int32))
        return strong_types(state), strong_types(metrics)

    def update(self, state):
        state, metrics = self._compiled_update(strong_types(state))
        metrics = jax.device_get(metrics)
        if not all(np.all(np.isfinite(a)) for a in metrics.values()):
            raise FloatingPointError("Nonfinite central development update")
        # A centralized reference has no packet link; it is not an 8-bit or
        # 520-bit operating point. None serializes as JSON null, never zero.
        metrics["communication_cost_bits"] = None
        return state, metrics

    def save_checkpoint(self, path, state):
        from .runtime import file_sha256
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        signatures = leaf_signatures(state)
        if any(a["weak_type"] for a in signatures):
            raise ValueError("Refusing a checkpoint with noncanonical weak array metadata")
        content = {"format": "centralized_ppo_v1", "config": asdict(self.config),
                   "implementation_sha256": self.implementation_sha256,
                   "env_class": type(self.env).__module__ + "." + type(self.env).__qualname__,
                   "env_dt": float(self.env.dt), "leaf_signatures": signatures,
                   "numerical_runtime": self.runtime,
                   "normalizer": {"kind": "fixed_identity", "updates": 0},
                   "scheduler": {"kind": "constant", "learning_rate": self.config.learning_rate},
                   "communication_cost": "not_applicable", "state": jax.device_get(state)}
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as stream:
            pickle.dump(content, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        return file_sha256(path)

    def load_checkpoint(self, path):
        with Path(path).open("rb") as stream:
            content = pickle.load(stream)
        if (content.get("format") != "centralized_ppo_v1" or content["config"] != asdict(self.config)
                or content["implementation_sha256"] != self.implementation_sha256
                or content.get("numerical_runtime") != self.runtime
                or content["env_dt"] != float(self.env.dt)
                or content["env_class"] != type(self.env).__module__ + "." + type(self.env).__qualname__):
            raise ValueError("Central checkpoint/config/implementation/environment mismatch")
        state = strong_types(jax.tree.map(jnp.asarray, content["state"]))
        if leaf_signatures(state) != content["leaf_signatures"]:
            raise ValueError("Restored array metadata differ from the saved canonical signature")
        return state


def load_central_config(path):
    config = CentralConfig(**json.loads(Path(path).read_text()))
    config.validate()
    return config
