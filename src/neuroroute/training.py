"""Development mixed-action PPO; no claim of a resolved confirmation protocol.

Checkpoints are Python pickle files of this program's own complete training
state. Load only trusted local checkpoints: pickle is not an interchange format
and may execute code. All JAX execution belongs in a compute allocation.
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
from flax import linen as nn
from flax import struct

from .communication import (
    FULL_BITS, bit_cost, cache_features, empty_state, full_gates, periodic_gates,
    reset_state,
)
from .policy import HybridPolicy


def implementation_fingerprint():
    digest = hashlib.sha256()
    for name in ("training.py", "policy.py", "communication.py", "environment.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainConfig:
    phase: str = "development"
    method: str = "B_FULL"
    seed: int = 0
    num_envs: int = 16
    horizon: int = 32
    num_updates: int = 4
    epochs: int = 2
    minibatch_size: int = 128
    history_length: int = 3
    episode_length: int = 1000
    local_width: int = 64
    global_width: int = 128
    critic_width: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_gate: float = 1e-3
    entropy_message: float = 1e-4
    entropy_motor: float = 1e-3
    max_grad_norm: float = 1.0
    budget_fraction: float = 1.0
    dual_learning_rate: float = 0.05
    lambda_bio: float = 0.0
    # (uplink fraction, downlink fraction), never inferred from robot scores.
    pi_bio: tuple = (0.5, 0.5)
    prior_manifest_sha256: str = ""
    prior_manifest_path: str = ""
    prior_readiness_sha256: str = ""
    prior_readiness_path: str = ""
    normalization: str = "fixed_identity"

    def validate(self):
        if self.phase != "development" or self.seed not in (0, 1):
            raise ValueError("This runner permits development seeds 0/1 only; confirmation is unresolved")
        allowed = {"B_FULL", "B_PERIODIC", "B_ASYM", "M_BIO", "B_SYM", "M_REVERSE"}
        if self.method not in allowed:
            raise ValueError(f"Method not implemented faithfully: {self.method}")
        for name in ("num_envs", "horizon", "num_updates", "epochs", "minibatch_size",
                     "episode_length", "local_width", "global_width", "critic_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_envs * self.horizon % self.minibatch_size:
            raise ValueError("Rollout size must be divisible by minibatch_size")
        if self.history_length != 3 or self.normalization != "fixed_identity":
            raise ValueError("Development implementation locks 3-frame history and fixed identity normalization")
        if self.budget_fraction not in (0.25, 0.5, 1.0):
            raise ValueError("Unsupported protocol budget")
        if self.method == "B_FULL" and self.budget_fraction != 1.0:
            raise ValueError("B_FULL requires its actual 100% cost budget")
        if self.method != "B_FULL" and self.budget_fraction == 1.0:
            raise ValueError("Budgeted methods require 25% or 50%")
        if self.lambda_bio < 0:
            raise ValueError("Prior weight must be nonnegative")
        if self.method in {"B_FULL", "B_PERIODIC", "B_ASYM"} and self.lambda_bio != 0:
            raise ValueError("This baseline cannot receive a biological prior")
        if self.method == "B_SYM" and self.lambda_bio <= 0:
            raise ValueError("B_SYM requires a declared positive symmetry penalty")
        if len(self.pi_bio) != 2 or min(self.pi_bio) <= 0 or not np.isclose(sum(self.pi_bio), 1):
            raise ValueError("pi_bio must be a positive normalized two-direction distribution")
        if self.method in {"M_BIO", "M_REVERSE"} and self.lambda_bio > 0:
            if len(self.prior_manifest_sha256) != 64 or any(
                    c not in "0123456789abcdef" for c in self.prior_manifest_sha256):
                raise ValueError("A real validated prior manifest SHA256 is required")
            manifest = Path(self.prior_manifest_path)
            if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != self.prior_manifest_sha256:
                raise ValueError("Prior manifest missing or checksum mismatch")
            prior = json.loads(manifest.read_text())
            if not prior.get("provenance") or not np.allclose(prior.get("pi_bio", []), self.pi_bio):
                raise ValueError("Prior probabilities/provenance disagree with validated source artifact")
            validate_prior_readiness(self, prior)
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("Invalid discount/GAE settings")


def validate_prior_readiness(config, prior):
    """A candidate prior file alone is insufficient for the biological arm."""
    path = Path(config.prior_readiness_path)
    if (not path.is_file() or len(config.prior_readiness_sha256) != 64
            or hashlib.sha256(path.read_bytes()).hexdigest() != config.prior_readiness_sha256):
        raise ValueError("Technical-readiness manifest missing or checksum mismatch")
    readiness = json.loads(path.read_text())
    labels = {"base"}
    labels.update(f"threshold_{t}_{normalization}" for t in (1, 5, 10)
                  for normalization in ("type_pair_mean_input_fraction", "cell_pair_mean_input_fraction"))
    labels.update(f"annotation_{i}" for i in range(6))
    variants = readiness.get("variants", [])
    if (readiness.get("status") != "technical_pass_candidate_mapping"
            or readiness.get("expected_variants") != 13
            or readiness.get("completed_variants") != 13
            or len(variants) != 13
            or {v.get("variant") for v in variants} != labels
            or readiness.get("errors") != [] or readiness.get("warnings") != []
            or readiness.get("all_direction_relations_agree") is not True):
        raise ValueError("Full hashed preprocessing/sensitivity technical audit has not passed")
    frozen = readiness.get("frozen_candidate", [])
    base = next(v for v in variants if v["variant"] == "base")
    if (len(frozen) != 2 or not np.allclose(frozen, config.pi_bio, rtol=0, atol=1e-12)
            or base.get("prior_sha256") != config.prior_manifest_sha256
            or prior.get("direction_order") != ["upward", "downward"]
            or not prior["provenance"].get("source_sha256")):
        raise ValueError("Frozen base prior, direction ordering, or source hashes disagree")


@struct.dataclass
class History:
    global_obs: jnp.ndarray  # [B,H,G]
    local_obs: jnp.ndarray   # [B,4,H,L]

    def actor_inputs(self):
        return (self.global_obs.reshape(self.global_obs.shape[:-2] + (-1,)),
                self.local_obs.reshape(self.local_obs.shape[:-2] + (-1,)))


def initialize_history(global_obs, local_obs, length=3):
    """Repeat the first permitted observation; never include a previous episode."""
    return History(jnp.repeat(global_obs[..., None, :], length, axis=-2),
                   jnp.repeat(local_obs[..., None, :], length, axis=-2))


def append_history(history, global_obs, local_obs):
    return History(jnp.concatenate((history.global_obs[..., 1:, :], global_obs[..., None, :]), axis=-2),
                   jnp.concatenate((history.local_obs[..., 1:, :], local_obs[..., None, :]), axis=-2))


def select_environments(mask, when_true, when_false):
    """Batch-local selection on arbitrary PyTrees of environment state."""
    return jax.tree.map(
        lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b),
        when_true, when_false,
    )


def critic_input(observation, history, cache):
    privileged = observation["privileged_state"]
    global_history, local_history = history.actor_inputs()
    batch = global_history.shape[:-1]
    return jnp.concatenate((privileged, global_history, local_history.reshape(batch + (-1,)),
                            cache_features(cache.up).reshape(batch + (-1,)),
                            cache_features(cache.down).reshape(batch + (-1,))), axis=-1)


class ValueNetwork(nn.Module):
    width: int = 128

    @nn.compact
    def __call__(self, observations):
        hidden = nn.tanh(nn.Dense(self.width)(observations))
        hidden = nn.tanh(nn.Dense(self.width)(hidden))
        # Shared trunk, identical privileged inputs, separate task/cost heads.
        return nn.Dense(2)(hidden)


@struct.dataclass
class TrainState:
    params: object
    optimizer_state: object
    env_state: object
    history: History
    cache: object
    rng: jnp.ndarray
    dual: jnp.ndarray
    update_index: jnp.ndarray
    environment_steps: jnp.ndarray
    episode_steps: jnp.ndarray


@struct.dataclass
class Transition:
    global_obs: jnp.ndarray
    local_obs: jnp.ndarray
    cache: object
    action: object
    forced_gates: jnp.ndarray
    old_log_prob: jnp.ndarray
    critic_obs: jnp.ndarray
    values: jnp.ndarray
    next_values: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    wire_bits: jnp.ndarray


def generalized_advantage(rewards, values, next_values, terminated, truncated,
                          gamma=0.99, gae_lambda=0.95):
    """[T,B,K] GAE: time limits bootstrap, either episode boundary stops traces."""
    bootstrap = jnp.where(terminated[..., None].astype(jnp.bool_), 0.0, next_values)
    delta = rewards + gamma * bootstrap - values
    continuation = 1.0 - jnp.maximum(terminated, truncated)

    def backward(carry, inputs):
        d, live = inputs
        advantage = d + gamma * gae_lambda * live[..., None] * carry
        return advantage, advantage

    _, advantages = jax.lax.scan(backward, jnp.zeros_like(delta[0]),
                                  (delta, continuation), reverse=True)
    return advantages, advantages + values


def allocation_penalty(gate_probs, pi, sample_weights=None):
    """KL of mean expected payload direction; fixed flag overhead excluded."""
    counts = jnp.stack((gate_probs[..., :4].sum(axis=-1), gate_probs[..., 4:].sum(axis=-1)), axis=-1)
    if sample_weights is not None:
        counts = counts * sample_weights[..., None]
    mean_counts = counts.reshape((-1, 2)).mean(axis=0)
    fractions = (mean_counts + 1e-8) / (mean_counts.sum() + 2e-8)
    prior = jnp.asarray(pi, dtype=fractions.dtype)
    return jnp.sum(fractions * (jnp.log(fractions) - jnp.log(prior))), fractions


def effective_prior(config):
    if config.method == "B_SYM":
        return (0.5, 0.5)
    if config.method == "M_REVERSE":
        return tuple(reversed(config.pi_bio))
    return tuple(config.pi_bio)


def ppo_objective(policy_output, predicted_values, batch, advantages, returns, dual, config):
    """Same objective code path for ASYM and BIO with lambda_bio=0."""
    combined = advantages[..., 0] - dual * advantages[..., 1]
    combined = (combined - combined.mean()) / (combined.std() + 1e-8)
    log_ratio = policy_output.log_prob - batch.old_log_prob
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    actor_loss = -jnp.minimum(ratio * combined, clipped * combined).mean()
    value_errors = 0.5 * jnp.square(predicted_values - returns).mean(axis=0)
    # Downlink expected gates depend on sampled uplink messages. Importance
    # weighting by the complete policy density includes that causal score
    # gradient; differentiating conditional gate scores alone would omit it.
    # Full-action weighting adds variance but does not require a hidden route
    # or an invalid derivative through a discrete packet.
    prior_loss, direction = allocation_penalty(policy_output.gate_probs, effective_prior(config), ratio)
    entropy_bonus = (config.entropy_gate * policy_output.entropy_gate.mean()
                     + config.entropy_message * policy_output.entropy_message.mean()
                     + config.entropy_motor * policy_output.entropy_motor.mean())
    total = (actor_loss + config.value_coefficient * value_errors.sum()
             - entropy_bonus + config.lambda_bio * prior_loss)
    metrics = {"loss": total, "actor_loss": actor_loss, "task_value_loss": value_errors[0],
               "cost_value_loss": value_errors[1], "prior_loss": prior_loss,
               "expected_up_fraction": direction[0], "entropy_bonus": entropy_bonus,
               "entropy_gate": policy_output.entropy_gate.mean(),
               "entropy_message": policy_output.entropy_message.mean(),
               "entropy_motor": policy_output.entropy_motor.mean(),
               "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
               "clip_fraction": (jnp.abs(ratio - 1.0) > config.clip_epsilon).mean()}
    return total, metrics


class PPOTrainer:
    def __init__(self, env, config: TrainConfig, *, split_observation=None, assemble_action=None):
        config.validate()
        self.env, self.config = env, config
        self.implementation_sha256 = implementation_fingerprint()
        if split_observation is None or assemble_action is None:
            from .environment import assemble_actions, split_observations
            split_observation = lambda obs: split_observations(obs, env.leg_map)
            assemble_action = lambda action: assemble_actions(action, env.leg_map)
        self.split_observation = split_observation
        self.assemble_action = assemble_action
        self.actor = HybridPolicy(config.local_width, config.global_width)
        self.critic = ValueNetwork(config.critic_width)
        self.optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(config.learning_rate, eps=1e-5))
        self._compiled_initialize = jax.jit(self._initialize)
        self._compiled_update = jax.jit(self._update)

    @property
    def scheduled(self):
        return self.config.method in {"B_FULL", "B_PERIODIC"}

    def schedule(self, episode_steps):
        if self.config.method == "B_FULL":
            return full_gates(episode_steps.shape)
        if self.config.method == "B_PERIODIC":
            return periodic_gates(episode_steps, self.config.budget_fraction)
        return jnp.zeros(episode_steps.shape + (8,), dtype=jnp.bool_)

    def initialize(self):
        return self._compiled_initialize(jax.random.PRNGKey(self.config.seed))

    def _initialize(self, key):
        key, env_key, actor_key, sample_key, critic_key = jax.random.split(key, 5)
        env_state = jax.vmap(self.env.reset)(jax.random.split(env_key, self.config.num_envs))
        global_obs, local_obs = self.split_observation(env_state.obs)
        history = initialize_history(global_obs, local_obs, self.config.history_length)
        cache = empty_state((self.config.num_envs,))
        global_hist, local_hist = history.actor_inputs()
        actor = self.actor.init(actor_key, global_hist, local_hist, cache, sample_key, self.env.dt)["params"]
        critic = self.critic.init(critic_key, critic_input(env_state.obs, history, cache))["params"]
        params = {"actor": actor, "critic": critic}
        return TrainState(params, self.optimizer.init(params), env_state, history, cache, key,
                          jnp.array(0.0), jnp.array(0, jnp.int32), jnp.array(0, jnp.int32),
                          jnp.zeros((self.config.num_envs,), jnp.int32))

    def _rollout_step(self, state, unused):
        key, actor_key, reset_key = jax.random.split(state.rng, 3)
        global_obs, local_obs = state.history.actor_inputs()
        gates = self.schedule(state.episode_steps)
        output = self.actor.apply({"params": state.params["actor"]}, global_obs, local_obs,
                                  state.cache, actor_key, self.env.dt,
                                  forced_gates=gates if self.scheduled else None)
        ci = critic_input(state.env_state.obs, state.history, state.cache)
        values = self.critic.apply({"params": state.params["critic"]}, ci)
        next_env = jax.vmap(self.env.step)(state.env_state, self.assemble_action(output.motor_action))
        next_history = append_history(state.history, *self.split_observation(next_env.obs))
        next_ci = critic_input(next_env.obs, next_history, output.cache)
        next_values = self.critic.apply({"params": state.params["critic"]}, next_ci)
        steps = state.episode_steps + 1
        terminated = next_env.done.astype(jnp.bool_)
        truncated = (steps >= self.config.episode_length) & ~terminated
        done = terminated | truncated
        costs = bit_cost(output.action.up_gate, output.action.down_gate)
        rewards = jnp.stack((next_env.reward, costs.total / FULL_BITS), axis=-1)
        transition = Transition(global_obs, local_obs, state.cache, output.action, gates,
                                output.log_prob, ci, values, next_values, rewards,
                                terminated.astype(jnp.float32), truncated.astype(jnp.float32),
                                jnp.stack((costs.total, costs.up, costs.down), axis=-1))

        def reset_ended(args):
            env_state, history, cache = args
            fresh = jax.vmap(self.env.reset)(jax.random.split(reset_key, self.config.num_envs))
            fresh_history = initialize_history(*self.split_observation(fresh.obs), self.config.history_length)
            return (select_environments(done, fresh, env_state),
                    select_environments(done, fresh_history, history), reset_state(cache, done))

        next_env, next_history, next_cache = jax.lax.cond(
            jnp.any(done), reset_ended, lambda args: args, (next_env, next_history, output.cache))
        return state.replace(env_state=next_env, history=next_history, cache=next_cache, rng=key,
                             episode_steps=jnp.where(done, 0, steps),
                             environment_steps=state.environment_steps + self.config.num_envs), transition

    def _update(self, state):
        state, rollout = jax.lax.scan(self._rollout_step, state, None, length=self.config.horizon)
        advantages, returns = generalized_advantage(
            rollout.rewards, rollout.values, rollout.next_values, rollout.terminated,
            rollout.truncated, self.config.gamma, self.config.gae_lambda)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        batch = jax.tree.map(flat, rollout)
        advantages, returns = flat(advantages), flat(returns)
        size = self.config.num_envs * self.config.horizon
        minibatches = size // self.config.minibatch_size

        def minibatch_step(carry, indices):
            params, optimizer_state, key = carry
            key, entropy_key = jax.random.split(key)
            selected = jax.tree.map(lambda a: a[indices], batch)

            def loss(params):
                output = self.actor.apply({"params": params["actor"]}, selected.global_obs,
                    selected.local_obs, selected.cache, entropy_key, self.env.dt,
                    action=selected.action,
                    forced_gates=selected.forced_gates if self.scheduled else None)
                predicted = self.critic.apply({"params": params["critic"]}, selected.critic_obs)
                return ppo_objective(output, predicted, selected, advantages[indices],
                                     returns[indices], state.dual, self.config)

            (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params)
            updates, optimizer_state = self.optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            metrics["gradient_norm"] = optax.global_norm(grads)
            return (params, optimizer_state, key), metrics

        def epoch_step(carry, unused):
            params, optimizer_state, key = carry
            key, permutation_key = jax.random.split(key)
            indices = jax.random.permutation(permutation_key, size).reshape((minibatches, self.config.minibatch_size))
            return jax.lax.scan(minibatch_step, (params, optimizer_state, key), indices)

        (params, optimizer_state, key), training_metrics = jax.lax.scan(
            epoch_step, (state.params, state.optimizer_state, state.rng), None, length=self.config.epochs)
        mean_cost = rollout.rewards[..., 1].mean()
        dual = jnp.maximum(0.0, state.dual + self.config.dual_learning_rate * (mean_cost - self.config.budget_fraction))
        if self.scheduled:
            dual = jnp.array(0.0)  # No learned scheduling constraint for fixed baselines.
        metrics = jax.tree.map(jnp.mean, training_metrics)
        costs = rollout.wire_bits[..., 0]
        payload_up = rollout.wire_bits[..., 1].sum() - 4 * size
        payload_total = costs.sum() - 8 * size
        metrics.update({"mean_task_reward": rollout.rewards[..., 0].mean(),
                        "mean_bits_per_step": costs.mean(), "total_bits": costs.sum(),
                        "up_bits": rollout.wire_bits[..., 1].sum(),
                        "down_bits": rollout.wire_bits[..., 2].sum(),
                        "bits_per_second": costs.sum() / (size * self.env.dt),
                        "bits_p50": jnp.quantile(costs, 0.5), "bits_p95": jnp.quantile(costs, 0.95),
                        "payload_up_fraction": payload_up / jnp.maximum(payload_total, 1),
                        "budget_excess_bits": costs.mean() - FULL_BITS * self.config.budget_fraction,
                        "budget_feasible": costs.mean() <= FULL_BITS * self.config.budget_fraction,
                        "terminated_episodes": rollout.terminated.sum(),
                        "truncated_episodes": rollout.truncated.sum(), "dual": dual,
                        "packet_gates": jnp.concatenate((rollout.action.up_gate, rollout.action.down_gate), axis=-1),
                        "raw_task_reward": rollout.rewards[..., 0],
                        "episode_terminated": rollout.terminated,
                        "episode_truncated": rollout.truncated})
        return state.replace(params=params, optimizer_state=optimizer_state, rng=key, dual=dual,
                             update_index=state.update_index + 1), metrics

    def update(self, state):
        state, metrics = self._compiled_update(state)
        # Block here so a caller cannot publish a checkpoint after an unnoticed
        # asynchronous divergence. No optimization step is hidden in evaluation.
        host_metrics = jax.device_get(metrics)
        if not all(np.all(np.isfinite(value)) for value in host_metrics.values()):
            raise FloatingPointError("Nonfinite development PPO update; preserve previous checkpoint")
        return state, host_metrics

    def save_checkpoint(self, path, state):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {"format_version": 1, "config": asdict(self.config), "env_dt": float(self.env.dt),
                   "implementation_sha256": self.implementation_sha256,
                   "environment_class": type(self.env).__module__ + "." + type(self.env).__qualname__,
                   "normalizer": {"kind": "fixed_identity", "updates": 0},
                   "scheduler": {"kind": "constant_learning_rate", "value": self.config.learning_rate},
                   "state": jax.device_get(state)}
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as stream:
            pickle.dump(content, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_checkpoint(self, path):
        """Read only a checkpoint created by this trusted local program."""
        with Path(path).open("rb") as stream:
            content = pickle.load(stream)
        if (content["format_version"] != 1 or content["config"] != asdict(self.config)
                or content["env_dt"] != float(self.env.dt)
                or content["implementation_sha256"] != self.implementation_sha256
                or content["environment_class"] != type(self.env).__module__ + "." + type(self.env).__qualname__):
            raise ValueError("Checkpoint/config/environment mismatch")
        return jax.tree.map(jnp.asarray, content["state"])


def load_config(path):
    values = json.loads(Path(path).read_text())
    if "pi_bio" in values:
        values["pi_bio"] = tuple(values["pi_bio"])
    config = TrainConfig(**values)
    config.validate()
    return config
