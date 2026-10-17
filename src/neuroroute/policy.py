"""Causal discrete-message/continuous-motor actor with exact joint likelihood.

No method identifier or biological prior is an actor input: B_ASYM and M_BIO
use this exact class. Their sole difference belongs in the training objective.
Entropy components are conditional entropies on the sampled causal contexts;
the payload terms are masked with the *actual* flags, as in the protocol.
"""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from .communication import CommunicationState, cache_features, deliver, tick


@struct.dataclass
class HybridAction:
    up_gate: jnp.ndarray
    up_payload: jnp.ndarray
    down_gate: jnp.ndarray
    down_payload: jnp.ndarray
    motor_raw: jnp.ndarray


@struct.dataclass
class PolicyOutput:
    action: HybridAction
    motor_action: jnp.ndarray
    cache: CommunicationState
    log_prob: jnp.ndarray
    entropy: jnp.ndarray
    gate_probs: jnp.ndarray
    entropy_gate: jnp.ndarray
    entropy_message: jnp.ndarray
    entropy_motor: jnp.ndarray
    log_prob_gate: jnp.ndarray
    log_prob_message: jnp.ndarray
    log_prob_motor: jnp.ndarray
    motor_mean: jnp.ndarray


def bernoulli_log_prob(logits, flags):
    return jnp.where(flags, -jax.nn.softplus(-logits), -jax.nn.softplus(logits))


def bernoulli_entropy(logits):
    prob = jax.nn.sigmoid(logits)
    return (1.0 - prob) * jax.nn.softplus(logits) + prob * jax.nn.softplus(-logits)


def categorical_terms(logits, symbols):
    logs = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(logs, symbols.astype(jnp.int32)[..., None], axis=-1)
    log_prob = selected[..., 0].sum(axis=-1)
    # Subtract the maximum first. This equivalent log-partition expression
    # avoids summing 256 small p*log(p) terms for uniform vocabularies, where
    # float32 accumulation otherwise fails the predeclared analytic tolerance.
    centered = logits - jnp.max(logits, axis=-1, keepdims=True)
    log_partition = jax.scipy.special.logsumexp(centered, axis=-1)
    entropy = (log_partition - (jnp.exp(logs) * centered).sum(axis=-1)).sum(axis=-1)
    return log_prob, entropy


def squashed_gaussian_log_prob(raw, mean, log_std):
    """Density of tanh(raw); saved pre-tanh samples avoid inverse instability."""
    gaussian = -0.5 * (((raw - mean) * jnp.exp(-log_std)) ** 2
                       + 2.0 * log_std + math.log(2.0 * math.pi))
    log_jacobian = 2.0 * (math.log(2.0) - raw - jax.nn.softplus(-2.0 * raw))
    return (gaussian - log_jacobian).sum(axis=(-1, -2))


class MLP(nn.Module):
    width: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.tanh(nn.Dense(self.width)(inputs))
        return nn.tanh(nn.Dense(self.width)(x))


class HybridPolicy(nn.Module):
    local_width: int = 64
    global_width: int = 128
    leg_embedding_dim: int = 8

    @nn.compact
    def __call__(
        self, global_obs, local_obs, cache, key, dt,
        *, action=None, forced_gates=None, deterministic_motor=False,
    ):
        """Sample, or teacher-force a recorded complete action.

        ``global_obs`` has shape [...,G], ``local_obs`` [...,4,L]. A single
        PRNG key samples all leading batch axes; vmap over keys also works.
        ``forced_gates`` [...,8] is an external public schedule (B_FULL or
        B_PERIODIC), whose gates have zero policy log-probability/entropy.
        Supply the same schedule during PPO recomputation. ``dt`` is seconds.
        Evaluation uses ``deterministic_motor=True`` while messages and gates
        remain sampled. Returned ``motor_action`` has shape [...,4,3] in [-1,1].
        """
        if local_obs.shape[-2] != 4:
            raise ValueError("local_obs must have exactly four legs")
        up_gate_key, up_msg_key, down_gate_key, down_msg_key, motor_key = jax.random.split(key, 5)
        batch_shape = local_obs.shape[:-2]
        leg_id = nn.Embed(4, self.leg_embedding_dim, name="leg_embedding")(jnp.arange(4))
        leg_id = jnp.broadcast_to(leg_id, batch_shape + leg_id.shape)
        aged = tick(cache, dt)

        # Each sender sees only its own private observation and downlink cache.
        local_input = jnp.concatenate((local_obs, leg_id, cache_features(aged.down)), axis=-1)
        local_hidden = MLP(self.local_width, name="local_encoder")(local_input)
        up_logits = nn.Dense(1, name="up_gate")(local_hidden)[..., 0]
        up_symbol_logits = nn.Dense(8 * 256, name="up_message")(local_hidden)
        up_symbol_logits = up_symbol_logits.reshape(batch_shape + (4, 8, 256))
        if action is None:
            up_gate = jax.random.bernoulli(up_gate_key, jax.nn.sigmoid(up_logits))
            if forced_gates is not None:
                up_gate = jnp.broadcast_to(forced_gates, batch_shape + (8,))[..., :4].astype(jnp.bool_)
            up_payload = jax.random.categorical(up_msg_key, up_symbol_logits).astype(jnp.uint8)
            up_payload = jnp.where(up_gate[..., None], up_payload, 0)
        else:
            up_gate, up_payload = action.up_gate, action.up_payload
        up_cache = deliver(aged.up, up_gate, up_payload)

        # Only delivered bytes can carry local information into the global MLP.
        global_input = jnp.concatenate(
            (global_obs, cache_features(up_cache).reshape(batch_shape + (-1,))), axis=-1,
        )
        global_hidden = MLP(self.global_width, name="global_encoder")(global_input)
        global_per_leg = jnp.broadcast_to(global_hidden[..., None, :], batch_shape + (4, self.global_width))
        down_hidden = MLP(self.local_width, name="down_encoder")(
            jnp.concatenate((global_per_leg, leg_id), axis=-1)
        )
        down_logits = nn.Dense(1, name="down_gate")(down_hidden)[..., 0]
        down_symbol_logits = nn.Dense(8 * 256, name="down_message")(down_hidden)
        down_symbol_logits = down_symbol_logits.reshape(batch_shape + (4, 8, 256))
        if action is None:
            down_gate = jax.random.bernoulli(down_gate_key, jax.nn.sigmoid(down_logits))
            if forced_gates is not None:
                down_gate = jnp.broadcast_to(forced_gates, batch_shape + (8,))[..., 4:].astype(jnp.bool_)
            down_payload = jax.random.categorical(down_msg_key, down_symbol_logits).astype(jnp.uint8)
            down_payload = jnp.where(down_gate[..., None], down_payload, 0)
        else:
            down_gate, down_payload = action.down_gate, action.down_payload
        down_cache = deliver(aged.down, down_gate, down_payload)

        # Raw global context is intentionally absent from the motor head.
        motor_hidden = MLP(self.local_width, name="motor_encoder")(
            jnp.concatenate((local_obs, leg_id, cache_features(down_cache)), axis=-1)
        )
        motor_mean = nn.Dense(3, kernel_init=nn.initializers.orthogonal(0.01), name="motor_mean")(motor_hidden)
        log_std = self.param("motor_log_std", nn.initializers.zeros, (3,))
        log_std = jnp.clip(log_std, -5.0, 2.0)
        if action is None:
            motor_raw = motor_mean if deterministic_motor else (
                motor_mean + jnp.exp(log_std) * jax.random.normal(motor_key, motor_mean.shape)
            )
        else:
            motor_raw = action.motor_raw

        all_logits = jnp.concatenate((up_logits, down_logits), axis=-1)
        all_gates = jnp.concatenate((up_gate, down_gate), axis=-1)
        if forced_gates is None:
            gate_log_prob = bernoulli_log_prob(all_logits, all_gates).sum(axis=-1)
            gate_entropy = bernoulli_entropy(all_logits).sum(axis=-1)
            gate_probs = jax.nn.sigmoid(all_logits)
        else:
            gate_log_prob = jnp.zeros(batch_shape)
            gate_entropy = jnp.zeros(batch_shape)
            gate_probs = jnp.broadcast_to(forced_gates, batch_shape + (8,)).astype(jnp.float32)
            # Impossible scheduled actions get zero density during replay.
            valid_schedule = jnp.all(all_gates == gate_probs.astype(jnp.bool_), axis=-1)
            gate_log_prob = jnp.where(valid_schedule, gate_log_prob, -jnp.inf)
        up_lp, up_ent = categorical_terms(up_symbol_logits, up_payload)
        down_lp, down_ent = categorical_terms(down_symbol_logits, down_payload)
        message_log_prob = (jnp.where(up_gate, up_lp, 0.0) + jnp.where(down_gate, down_lp, 0.0)).sum(axis=-1)
        message_entropy = (jnp.where(up_gate, up_ent, 0.0) + jnp.where(down_gate, down_ent, 0.0)).sum(axis=-1)
        motor_log_prob = squashed_gaussian_log_prob(motor_raw, motor_mean, log_std)
        # H[tanh(X)] = H[X] + E[log|J_tanh(X)|]. Draw a fresh, pathwise
        # current-policy sample for entropy, including during action replay.
        # This avoids treating rollout actions as current-policy samples or
        # silently losing the entropy gradient through tanh's Jacobian.
        base_motor_entropy = jnp.broadcast_to(
            4.0 * jnp.sum(log_std + 0.5 * math.log(2.0 * math.pi * math.e)), batch_shape,
        )
        entropy_raw = motor_mean + jnp.exp(log_std) * jax.random.normal(
            jax.random.fold_in(motor_key, 1), motor_mean.shape,
        )
        log_jacobian = 2.0 * (math.log(2.0) - entropy_raw - jax.nn.softplus(-2.0 * entropy_raw))
        motor_entropy = base_motor_entropy + log_jacobian.sum(axis=(-1, -2))
        joint = gate_log_prob + message_log_prob + motor_log_prob
        return PolicyOutput(
            HybridAction(up_gate, up_payload, down_gate, down_payload, motor_raw),
            jnp.tanh(motor_raw), CommunicationState(up_cache, down_cache), joint,
            gate_entropy + message_entropy + motor_entropy, gate_probs,
            gate_entropy, message_entropy, motor_entropy,
            gate_log_prob, message_log_prob, motor_log_prob, motor_mean,
        )
