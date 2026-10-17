"""Recursive MATCH v4 candidate: stable float32 normalization and density reductions.

The original paper's Q/TD objective is still replaced by joint PPO. History
contains actual past model reconstructions, and actor replay must reconstruct
that entire causal history under current parameters from episode reset.
"""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .communication import CommunicationState, deliver, tick
from .policy import (HybridAction, PolicyOutput, bernoulli_entropy,
    bernoulli_log_prob, categorical_terms, squashed_gaussian_log_prob)
from .published_policy import EntityPool


from .recurrent_published_policy import (RecurrentMatchOutput, RecurrentMatchState,
    decoded, empty_recurrent_match_state, reset_recurrent_match_state)



# Explicit float32 materialization and fixed binary reductions prevent the
# sampled and teacher-forced graphs from reassociating probability sums.
def materialize(x):
    return jax.lax.optimization_barrier(x)


def stable_sum(x):
    n = x.shape[-1]
    size = 1 << (n - 1).bit_length()
    if size != n:
        x = jnp.pad(x, [(0, 0)]*(x.ndim-1)+[(0, size-n)])
    while x.shape[-1] > 1:
        x = materialize(x[..., 0::2]+x[..., 1::2])
    return x[..., 0]


class LayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Centered variance implements the same normalization equation with
        # less cancellation than E[x*x]-E[x]**2. No rounding or quantization.
        mean = materialize(stable_sum(x)/x.shape[-1])
        centered = materialize(x-mean[..., None])
        variance = materialize(stable_sum(materialize(centered*centered))/x.shape[-1])
        inv = materialize(jax.lax.rsqrt(materialize(variance+1e-6)))
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        bias = self.param('bias', nn.initializers.zeros, (x.shape[-1],))
        normalized = materialize(centered*inv[..., None])
        return materialize(materialize(normalized*scale)+bias)


def categorical_terms(logits, symbols):
    centered = materialize(logits-materialize(jnp.max(logits, axis=-1, keepdims=True)))
    partition = materialize(jnp.log(stable_sum(materialize(jnp.exp(centered)))))
    logs = materialize(centered-partition[..., None])
    selected = materialize(jnp.take_along_axis(logs, symbols[..., None].astype(jnp.int32), -1)[..., 0])
    entropy = materialize(partition-stable_sum(materialize(jnp.exp(logs)*centered)))
    return stable_sum(selected), stable_sum(entropy)


def squashed_gaussian_log_prob(raw, mean, log_std):
    standardized = materialize(materialize(raw-mean)*materialize(jnp.exp(-log_std)))
    gaussian = materialize(-.5*materialize(materialize(standardized**2)+2*log_std+math.log(2*math.pi)))
    jac = materialize(2*materialize(math.log(2)-raw-jax.nn.softplus(-2*raw)))
    terms = materialize(gaussian-jac)
    return stable_sum(terms.reshape(terms.shape[:-2]+(-1,)))


class TransformerBlock(nn.Module):
    width: int
    heads: int

    @nn.compact
    def __call__(self, tokens, mask):
        hidden = LayerNorm()(tokens)
        hidden = nn.SelfAttention(num_heads=self.heads, qkv_features=self.width,
                                  out_features=self.width, dropout_rate=0.0)(
                                      hidden, mask=mask, deterministic=True)
        tokens = materialize(tokens + materialize(hidden))
        hidden = nn.Dense(self.width * 2)(LayerNorm()(tokens))
        hidden = nn.gelu(hidden)
        return materialize(tokens + materialize(nn.Dense(self.width)(hidden)))


class LocalRecurrentEncoder(nn.Module):
    width: int
    heads: int
    layers: int

    @nn.compact
    def __call__(self, observations, previous_hidden, receiver):
        # Each leg is an independent batch element, never an attention token.
        joints = observations.reshape(observations.shape[:-1] + (3, 3))
        # local observation is [q1..q3,dq1..dq3,a1..a3].
        joints = jnp.swapaxes(joints, -1, -2)
        joint_features = nn.Dense(self.width)(joints)
        identity = self.param("joint_position", nn.initializers.normal(.02), (3, self.width))
        joint_features = joint_features + identity
        reply = jnp.concatenate((decoded(receiver.payload, receiver.valid),
            receiver.valid[..., None], jnp.log1p(receiver.age)[..., None]), axis=-1)
        reply = nn.Dense(self.width, name="received_reply_embedding")(reply)
        tokens = jnp.concatenate((previous_hidden[..., None, :], joint_features, reply[..., None, :]), axis=-2)
        valid = jnp.ones(tokens.shape[:-1], jnp.bool_)
        mask = nn.make_attention_mask(valid, valid, dtype=jnp.bool_)
        for layer in range(self.layers):
            tokens = TransformerBlock(self.width, self.heads, name=f"block_{layer}")(tokens, mask)
        return LayerNorm()(tokens[..., 0, :])


class RecurrentPublishedPolicy(nn.Module):
    width: int = 32
    heads: int = 4
    layers: int = 1

    @nn.compact
    def __call__(self, global_obs, local_obs, state, key, dt, *, action=None,
                 up_received=None, down_received=None, deterministic_motor=False):
        if local_obs.shape[-2:] != (4, 9) or global_obs.shape[-1] != 12:
            raise ValueError("Go1 MATCH requires four private 9D views and 12D hub view")
        batch = global_obs.shape[:-1]
        length = state.history_valid.shape[-1]
        ku, kmu, kd, kmd, ka = jax.random.split(key, 5)
        aged = tick(state.cache, dt)
        local_encoder = LocalRecurrentEncoder(self.width, self.heads, self.layers, name="local_recurrent")
        # Numeric boundary only: preserve the same float32 value and derivative.
        # Sampling and teacher forcing must consume the same materialized local
        # recurrent output rather than fuse its normalization into later uses.
        hidden = jax.lax.optimization_barrier(
            local_encoder(local_obs, state.local_hidden, aged.down))
        up_logits = materialize(nn.Dense(1, name="up_gate")(hidden)[..., 0])
        obs_logits = nn.Dense(4 * 256, name="observation_symbols")(hidden).reshape(batch + (4, 4, 256))
        query_logits = nn.Dense(4 * 256, name="query_symbols")(hidden).reshape(batch + (4, 4, 256))
        up_symbols = materialize(jnp.concatenate((obs_logits, query_logits), axis=-2))
        if action is None:
            gu = jax.random.bernoulli(ku, jax.nn.sigmoid(up_logits))
            pu = jax.random.categorical(kmu, up_symbols).astype(jnp.uint8)
            pu = jnp.where(gu[..., None], pu, 0)
        else:
            gu, pu = action.up_gate, action.up_payload
        received_u = gu if up_received is None else gu & up_received
        up_cache = deliver(aged.up, received_u, pu)

        # A sole legal viewpoint per leg is a declared degenerate entity pool.
        views = jnp.concatenate((decoded(up_cache.payload[..., :4], up_cache.valid),
            up_cache.valid[..., None], jnp.log1p(up_cache.age)[..., None]), axis=-1)
        legs = EntityPool(self.width, name="entity_pool")(views[..., None, :], up_cache.valid[..., None])
        torso = nn.Dense(self.width, name="torso_encoder")(global_obs)
        current = jnp.concatenate((legs, torso[..., None, :]), axis=-2)
        prior = nn.Dense(self.width, name="prediction_history_encoder")(state.reconstructed_history)
        tokens = jnp.concatenate((prior, current[..., None, :, :]), axis=-3)
        temporal = self.param("hub_time_embedding", nn.initializers.normal(.02), (length + 1, 1, self.width))
        entity = self.param("hub_entity_embedding", nn.initializers.normal(.02), (1, 5, self.width))
        tokens = (tokens + temporal + entity).reshape(batch + ((length + 1) * 5, self.width))
        valid = jnp.concatenate((state.history_valid, jnp.ones(batch + (1,), jnp.bool_)), axis=-1)
        valid = jnp.repeat(valid, 5, axis=-1)
        mask = nn.make_attention_mask(valid, valid, dtype=jnp.bool_)
        for layer in range(self.layers):
            tokens = TransformerBlock(self.width, self.heads, name=f"hub_transformer_{layer}")(tokens, mask)
        reconstructed = LayerNorm(name="hub_output_norm")(tokens[..., -5:, :])
        positions = nn.Dense(3, name="position_reconstruction")(reconstructed)
        attributes = nn.Dense(6, name="attribute_reconstruction")(reconstructed)
        prediction = jnp.concatenate((positions, attributes), axis=-1)

        query = nn.Dense(self.width, name="query_decoder")(decoded(up_cache.payload[..., 4:], up_cache.valid))
        query = jnp.where(up_cache.valid[..., None], query, 0.)
        keys = nn.Dense(self.width, name="reply_keys")(reconstructed)
        scores = jnp.einsum("...id,...jd->...ij", query, keys) / math.sqrt(self.width)
        relative = positions[..., :4, None, :] - positions[..., None, :, :]
        distance = jnp.sqrt(jnp.sum(relative ** 2, axis=-1) + 1e-8)
        distance = distance / (distance.mean(axis=(-1, -2), keepdims=True) + 1e-6)
        gain = jax.nn.softplus(self.param("distance_attention_gain", nn.initializers.zeros, ()))
        attention = jax.nn.softmax(gain * scores * jax.nn.sigmoid(-distance), axis=-1)
        reply = jnp.einsum("...ij,...jd->...id", attention, reconstructed)
        reply = nn.tanh(nn.Dense(self.width, name="reply_encoder")(jnp.concatenate((reply, query), axis=-1)))
        down_logits = materialize(nn.Dense(1, name="down_gate")(reply)[..., 0])
        down_symbols = materialize(nn.Dense(8 * 256, name="reply_symbols")(reply).reshape(batch + (4, 8, 256)))
        if action is None:
            gd = jax.random.bernoulli(kd, jax.nn.sigmoid(down_logits))
            pd = jax.random.categorical(kmd, down_symbols).astype(jnp.uint8)
            pd = jnp.where(gd[..., None], pd, 0)
        else:
            gd, pd = action.down_gate, action.down_payload
        received_d = gd if down_received is None else gd & down_received
        down_cache = deliver(aged.down, received_d, pd)
        motor_hidden = local_encoder(local_obs, state.local_hidden, down_cache)
        mean = nn.Dense(3, kernel_init=nn.initializers.orthogonal(.01), name="motor_mean")(motor_hidden)
        log_std = jnp.clip(self.param("motor_log_std", nn.initializers.zeros, (3,)), -5., 2.)
        raw = action.motor_raw if action is not None else (
            mean if deterministic_motor else mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape))
        logits = jnp.concatenate((up_logits, down_logits), axis=-1)
        gates = jnp.concatenate((gu, gd), axis=-1)
        gate_lp = stable_sum(materialize(bernoulli_log_prob(materialize(logits), gates)))
        gate_entropy = bernoulli_entropy(logits).sum(axis=-1)
        uplp, upent = categorical_terms(up_symbols, pu)
        downlp, downent = categorical_terms(down_symbols, pd)
        message_lp = stable_sum(materialize(jnp.where(gu, uplp, 0.) + jnp.where(gd, downlp, 0.)))
        message_entropy = (jnp.where(gu, upent, 0.) + jnp.where(gd, downent, 0.)).sum(axis=-1)
        motor_lp = squashed_gaussian_log_prob(raw, mean, log_std)
        entropy_raw = mean + jnp.exp(log_std) * jax.random.normal(jax.random.fold_in(ka, 1), mean.shape)
        log_jac = 2. * (math.log(2.) - entropy_raw - jax.nn.softplus(-2. * entropy_raw))
        motor_entropy = 4 * jnp.sum(log_std + .5 * math.log(2 * math.pi * math.e)) + log_jac.sum(axis=(-1, -2))
        cache = CommunicationState(up_cache, down_cache)
        policy = PolicyOutput(HybridAction(gu, pu, gd, pd, raw), jnp.tanh(raw), cache,
            stable_sum(jnp.stack((gate_lp, message_lp, motor_lp), axis=-1)), gate_entropy + message_entropy + motor_entropy,
            jax.nn.sigmoid(logits), gate_entropy, message_entropy, motor_entropy,
            gate_lp, message_lp, motor_lp, mean)
        next_state = RecurrentMatchState(cache, hidden,
            jnp.concatenate((state.reconstructed_history[..., 1:, :, :], prediction[..., None, :, :]), axis=-3),
            jnp.concatenate((state.history_valid[..., 1:], jnp.ones(batch + (1,), jnp.bool_)), axis=-1))
        return RecurrentMatchOutput(policy, next_state, positions, attributes, attention)
