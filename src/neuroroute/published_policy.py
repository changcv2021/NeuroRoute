"""MATCH finite-history packet/PPO adaptation; original-task replication pending.

Preserves entity pooling, temporal reconstruction, uploaded queries,
distance-aware replies, local history attention, and an external reconstruction
loss. Replaces the original Q/TD learner with the project's joint-action PPO
interface. Raw histories are checkpointed and re-encoded under current weights;
this finite-window reconstruction differs from the paper's recursive buffer of
learned reconstructions. No true simulation state is an actor argument.
"""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from .communication import CommunicationState, cache_features, deliver, empty_state, tick
from .policy import (HybridAction, PolicyOutput, bernoulli_entropy,
                     bernoulli_log_prob, categorical_terms,
                     squashed_gaussian_log_prob)


@struct.dataclass
class MatchState:
    cache: CommunicationState
    up_history: jnp.ndarray          # [..., H, 4, 8], actual received events
    up_received: jnp.ndarray         # [..., H, 4], not sender logits or gates
    global_history: jnp.ndarray      # [..., H, 12], hub-owned observations
    local_history: jnp.ndarray       # [..., H, 4, 9], isolated at each leg
    down_history: jnp.ndarray        # [..., H, 4, 8], local receiver snapshots
    down_valid: jnp.ndarray          # [..., H, 4]
    down_age: jnp.ndarray            # [..., H, 4], local clock only
    frame_valid: jnp.ndarray         # [..., H], episode-boundary padding


@struct.dataclass
class MatchOutput:
    policy: PolicyOutput
    state: MatchState
    positions: jnp.ndarray           # [..., 5, 3], predictions, never labels
    attributes: jnp.ndarray          # [..., 5, 6], predicted q/dq
    reply_attention: jnp.ndarray     # [..., 4, 5]


@struct.dataclass
class ReconstructionTargets:
    positions: jnp.ndarray
    attributes: jnp.ndarray
    position_mask: jnp.ndarray
    attribute_mask: jnp.ndarray


def empty_match_state(batch_shape=(), history_length=4):
    if history_length < 1:
        raise ValueError("history_length must be positive")
    prefix = tuple(batch_shape) + (history_length,)
    return MatchState(
        empty_state(batch_shape),
        jnp.zeros(prefix + (4, 8), jnp.uint8),
        jnp.zeros(prefix + (4,), jnp.bool_),
        jnp.zeros(prefix + (12,), jnp.float32),
        jnp.zeros(prefix + (4, 9), jnp.float32),
        jnp.zeros(prefix + (4, 8), jnp.uint8),
        jnp.zeros(prefix + (4,), jnp.bool_),
        jnp.zeros(prefix + (4,), jnp.float32),
        jnp.zeros(prefix, jnp.bool_),
    )


def reset_match_state(state, reset_mask):
    """Clear selected episodes, including every historical private byte."""
    mask = jnp.asarray(reset_mask, jnp.bool_)
    return jax.tree.map(
        lambda value: jnp.where(mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim)),
                               jnp.zeros_like(value), value), state)


def privileged_targets(joint_position, joint_velocity, foot_position):
    """Training-only labels; caller supplies leg order FR,FL,RR,RL.

    q/dq: [...,4,3]; feet: [...,4,3] relative to torso. Torso position is the
    declared coordinate origin; torso q/dq do not exist and are masked out.
    The returned struct is accepted by reconstruction_loss, never by the actor.
    """
    if joint_position.shape[-2:] != (4, 3) or joint_velocity.shape != joint_position.shape:
        raise ValueError("joint labels must have shape [...,4,3]")
    if foot_position.shape != joint_position.shape:
        raise ValueError("foot labels must match joint label shape")
    prefix = joint_position.shape[:-2]
    positions = jnp.concatenate((foot_position, jnp.zeros(prefix + (1, 3))), axis=-2)
    attributes = jnp.concatenate((jnp.concatenate((joint_position, joint_velocity), axis=-1),
                                  jnp.zeros(prefix + (1, 6))), axis=-2)
    attr_mask = jnp.broadcast_to(jnp.arange(5)[..., None] < 4, attributes.shape)
    return ReconstructionTargets(positions, attributes, jnp.ones_like(positions, dtype=jnp.bool_), attr_mask)


def reconstruction_loss(positions, attributes, targets, *, position_scale=1.0,
                        attribute_scale=1.0):
    """Per-sample centered-position and q/dq MSE, for an external auxiliary loss.

    Scales must be positive constants from the frozen training normalization.
    No gradient is passed into privileged labels. Missing labels must be masked;
    NaNs in masked fields are neutralized before arithmetic.
    """
    pm = targets.position_mask.astype(positions.dtype)
    am = targets.attribute_mask.astype(attributes.dtype)
    truth_p = jax.lax.stop_gradient(jnp.where(pm > 0, targets.positions, 0.0))
    truth_a = jax.lax.stop_gradient(jnp.where(am > 0, targets.attributes, 0.0))
    count_p = jnp.maximum(pm.sum(axis=-2, keepdims=True), 1)
    center_p = (positions * pm).sum(axis=-2, keepdims=True) / count_p
    center_t = truth_p.sum(axis=-2, keepdims=True) / count_p
    error_p = ((positions - center_p) - (truth_p - center_t)) / jnp.asarray(position_scale)
    error_a = (attributes - truth_a) / jnp.asarray(attribute_scale)
    pos = jnp.where(pm > 0, error_p ** 2, 0).sum(axis=(-1, -2)) / jnp.maximum(pm.sum(axis=(-1, -2)), 1)
    attr = jnp.where(am > 0, error_a ** 2, 0).sum(axis=(-1, -2)) / jnp.maximum(am.sum(axis=(-1, -2)), 1)
    return {"position": pos, "attribute": attr, "total": pos + attr}


def _append(history, current, axis):
    """Append one genuine time sample, discarding the oldest sample."""
    joined = jnp.concatenate((history, jnp.expand_dims(current, axis=axis)), axis=axis)
    return jnp.take(joined, jnp.arange(1, joined.shape[axis]), axis=axis)


def _bytes(payload, valid):
    decoded = payload.astype(jnp.float32) / 127.5 - 1.0
    return jnp.where(valid[..., None], decoded, 0.0)


class TransformerBlock(nn.Module):
    width: int
    heads: int

    @nn.compact
    def __call__(self, tokens, mask):
        hidden = nn.LayerNorm()(tokens)
        hidden = nn.SelfAttention(num_heads=self.heads, qkv_features=self.width,
                                  out_features=self.width, dropout_rate=0.0)(
                                      hidden, mask=mask, deterministic=True)
        tokens = tokens + hidden
        hidden = nn.Dense(self.width * 2)(nn.LayerNorm()(tokens))
        hidden = nn.gelu(hidden)
        return tokens + nn.Dense(self.width)(hidden)


class EntityPool(nn.Module):
    """Masked view pooling; Go1 ownership admits one view per leg entity."""
    width: int

    @nn.compact
    def __call__(self, views, available):
        values = nn.Dense(self.width, name="value")(views)
        scores = nn.Dense(1, name="score")(nn.tanh(values))[..., 0]
        weights = jax.nn.softmax(jnp.where(available, scores, -1e9), axis=-1)
        weights = jnp.where(available, weights, 0.0)
        return (weights[..., None] * values).sum(axis=-2)


class LocalHistoryEncoder(nn.Module):
    width: int
    heads: int
    layers: int

    @nn.compact
    def __call__(self, observations, down_payload, down_valid, down_age, frame_valid):
        # Input time precedes leg: [...,T,4,...]. Exchange axes so attention
        # treats each leg as an independent batch member, never as a token.
        received = jnp.concatenate((_bytes(down_payload, down_valid),
                                    down_valid[..., None].astype(jnp.float32),
                                    jnp.log1p(down_age)[..., None]), axis=-1)
        raw = jnp.concatenate((observations, received), axis=-1)
        raw = jnp.swapaxes(raw, -3, -2)
        # Ignore arbitrary padding contents, which can otherwise leak via bias.
        mask_valid = frame_valid[..., None, :, None]
        tokens = nn.Dense(self.width)(jnp.where(mask_valid, raw, 0.0))
        length = raw.shape[-2]
        time = self.param("time_embedding", nn.initializers.normal(0.02), (length, self.width))
        leg = self.param("leg_embedding", nn.initializers.normal(0.02), (4, self.width))
        tokens = tokens + time + leg[..., None, :]
        mask = frame_valid[..., None, None, None, :]  # batch,leg,head,query,key
        for i in range(self.layers):
            tokens = TransformerBlock(self.width, self.heads, name=f"block_{i}")(tokens, mask)
        return nn.LayerNorm()(tokens[..., -1, :])


class PublishedMatchPolicy(nn.Module):
    """Faithful structural adaptation, not a claim of original MATCH parity."""
    width: int = 64
    heads: int = 4
    layers: int = 2

    @nn.compact
    def __call__(self, global_obs, local_obs, state, key, dt, *, action=None,
                 forced_gates=None, deterministic_motor=False,
                 up_received=None, down_received=None):
        """Receive masks optionally model externally supplied link delivery.

        Costs/likelihood always use sent action gates. Receivers see only
        ``gate & received`` packets. Replaying a rollout must use the same
        recorded delivery masks and pre-action MatchState. Ages are seconds.
        """
        if local_obs.shape[-2:] != (4, 9) or global_obs.shape[-1] != 12:
            raise ValueError("MATCH ownership schema requires local [...,4,9], global [...,12]")
        if local_obs.shape[:-2] != global_obs.shape[:-1]:
            raise ValueError("global/local batch shapes differ")
        if self.width % self.heads or self.layers < 1:
            raise ValueError("width must divide heads and at least one Transformer layer is required")
        batch = local_obs.shape[:-2]
        history_length = state.frame_valid.shape[-1]
        ku, kmu, kd, kmd, ka = jax.random.split(key, 5)
        aged = tick(state.cache, dt)
        frame_valid = jnp.concatenate((state.frame_valid, jnp.ones(batch + (1,), jnp.bool_)), axis=-1)
        local_frames = jnp.concatenate((state.local_history, local_obs[..., None, :, :]), axis=-3)
        local_encoder = LocalHistoryEncoder(self.width, self.heads, self.layers, name="local_history_transformer")

        def local_context(cache):
            payload = jnp.concatenate((state.down_history, cache.payload[..., None, :, :]), axis=-3)
            valid = jnp.concatenate((state.down_valid, cache.valid[..., None, :]), axis=-2)
            age = jnp.concatenate((state.down_age, cache.age[..., None, :]), axis=-2)
            return local_encoder(local_frames, payload, valid, age, frame_valid)

        local_hidden = local_context(aged.down)
        up_logits = nn.Dense(1, name="up_gate")(local_hidden)[..., 0]
        # Distinct learned encoders expose the compulsory 4+4 byte allocation.
        obs_logits = nn.Dense(4 * 256, name="observation_symbols")(local_hidden).reshape(batch + (4, 4, 256))
        query_logits = nn.Dense(4 * 256, name="query_symbols")(local_hidden).reshape(batch + (4, 4, 256))
        up_symbols = jnp.concatenate((obs_logits, query_logits), axis=-2)
        schedule = None if forced_gates is None else jnp.broadcast_to(forced_gates, batch + (8,)).astype(jnp.bool_)
        if action is None:
            gu = jax.random.bernoulli(ku, jax.nn.sigmoid(up_logits)) if schedule is None else schedule[..., :4]
            pu = jax.random.categorical(kmu, up_symbols).astype(jnp.uint8)
            pu = jnp.where(gu[..., None], pu, 0)
        else:
            gu, pu = action.up_gate, action.up_payload
        received_u = gu if up_received is None else gu & jnp.asarray(up_received, jnp.bool_)
        up_cache = deliver(aged.up, received_u, pu)

        # History contains actual byte arrival events at their real times.
        # Missing slots are zeroed; no repetition of today's cache into the past.
        current_event = jnp.where(received_u[..., None], pu, 0)
        events = jnp.concatenate((state.up_history, current_event[..., None, :, :]), axis=-3)
        event_valid = jnp.concatenate((state.up_received, received_u[..., None, :]), axis=-2)
        event_valid = event_valid & frame_valid[..., :, None]
        observation_values = _bytes(events[..., :4], event_valid)
        leg_tokens = EntityPool(self.width, name="entity_pool")(
            observation_values[..., None, :], event_valid[..., None])
        global_frames = jnp.concatenate((state.global_history, global_obs[..., None, :]), axis=-2)
        global_frames = jnp.where(frame_valid[..., None], global_frames, 0.0)
        torso_tokens = nn.Dense(self.width, name="torso_encoder")(global_frames)
        tokens = jnp.concatenate((leg_tokens, torso_tokens[..., None, :]), axis=-2)
        time_pos = self.param("hub_time_embedding", nn.initializers.normal(0.02), (history_length + 1, self.width))
        entity_pos = self.param("hub_entity_embedding", nn.initializers.normal(0.02), (5, self.width))
        tokens = tokens + time_pos[..., None, :] + entity_pos
        # A receiver's current cache supplies legitimate long-held data older
        # than the finite event window, but only at the current time token.
        cached_obs = _bytes(up_cache.payload[..., :4], up_cache.valid)
        cached_features = jnp.concatenate((cached_obs, up_cache.valid[..., None],
                                            jnp.log1p(up_cache.age)[..., None]), axis=-1)
        cache_token = nn.Dense(self.width, name="current_cache_encoder")(cached_features)
        tokens = tokens.at[..., -1, :4, :].add(cache_token)
        total_tokens = (history_length + 1) * 5
        tokens = tokens.reshape(batch + (total_tokens, self.width))
        times = jnp.repeat(jnp.arange(history_length + 1), 5)
        temporal_mask = times[:, None] >= times[None, :]
        key_valid = jnp.repeat(frame_valid, 5, axis=-1)
        mask = temporal_mask & key_valid[..., None, :]
        for i in range(self.layers):
            tokens = TransformerBlock(self.width, self.heads, name=f"hub_transformer_{i}")(
                tokens, mask[..., None, :, :])
        reconstructed = nn.LayerNorm(name="hub_output_norm")(tokens[..., -5:, :])
        positions = nn.Dense(3, name="position_reconstruction")(reconstructed)
        attributes = nn.Dense(6, name="attribute_reconstruction")(reconstructed)

        # Hub query is decoded only from received bytes; no local hidden state
        # is accessible in this branch, including under teacher forcing.
        query = nn.Dense(self.width, name="query_decoder")(_bytes(up_cache.payload[..., 4:], up_cache.valid))
        query = jnp.where(up_cache.valid[..., None], query, 0.0)
        keys = nn.Dense(self.width, name="reply_keys")(reconstructed)
        similarity = jnp.einsum("...id,...jd->...ij", query, keys) / math.sqrt(self.width)
        delta = positions[..., :4, None, :] - positions[..., None, :, :]
        distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-8)
        # Distances come from model predictions, never simulator geometry.
        distance = distance / (jnp.mean(distance, axis=(-1, -2), keepdims=True) + 1e-6)
        gain = jax.nn.softplus(self.param("distance_attention_gain", nn.initializers.zeros, ()))
        attention = jax.nn.softmax(gain * similarity * jax.nn.sigmoid(-distance), axis=-1)
        reply = jnp.einsum("...ij,...jd->...id", attention, reconstructed)
        reply_hidden = nn.tanh(nn.Dense(self.width, name="reply_encoder")(jnp.concatenate((reply, query), axis=-1)))
        down_logits = nn.Dense(1, name="down_gate")(reply_hidden)[..., 0]
        down_symbols = nn.Dense(8 * 256, name="reply_symbols")(reply_hidden).reshape(batch + (4, 8, 256))
        if action is None:
            gd = jax.random.bernoulli(kd, jax.nn.sigmoid(down_logits)) if schedule is None else schedule[..., 4:]
            pd = jax.random.categorical(kmd, down_symbols).astype(jnp.uint8)
            pd = jnp.where(gd[..., None], pd, 0)
        else:
            gd, pd = action.down_gate, action.down_payload
        received_d = gd if down_received is None else gd & jnp.asarray(down_received, jnp.bool_)
        down_cache = deliver(aged.down, received_d, pd)
        motor_hidden = local_context(down_cache)
        mean = nn.Dense(3, kernel_init=nn.initializers.orthogonal(0.01), name="motor_mean")(motor_hidden)
        log_std = jnp.clip(self.param("motor_log_std", nn.initializers.zeros, (3,)), -5.0, 2.0)
        raw = action.motor_raw if action is not None else (
            mean if deterministic_motor else mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape))
        logits = jnp.concatenate((up_logits, down_logits), axis=-1)
        gates = jnp.concatenate((gu, gd), axis=-1)
        if schedule is None:
            gate_lp = bernoulli_log_prob(logits, gates).sum(axis=-1)
            gate_ent = bernoulli_entropy(logits).sum(axis=-1)
            gate_probs = jax.nn.sigmoid(logits)
        else:
            gate_lp = jnp.where(jnp.all(gates == schedule, axis=-1), 0.0, -jnp.inf)
            gate_ent = jnp.zeros(batch)
            gate_probs = schedule.astype(jnp.float32)
        uplp, upent = categorical_terms(up_symbols, pu)
        downlp, downent = categorical_terms(down_symbols, pd)
        msg_lp = (jnp.where(gu, uplp, 0.0) + jnp.where(gd, downlp, 0.0)).sum(axis=-1)
        msg_ent = (jnp.where(gu, upent, 0.0) + jnp.where(gd, downent, 0.0)).sum(axis=-1)
        motor_lp = squashed_gaussian_log_prob(raw, mean, log_std)
        entropy_raw = mean + jnp.exp(log_std) * jax.random.normal(jax.random.fold_in(ka, 1), mean.shape)
        log_jac = 2.0 * (math.log(2.0) - entropy_raw - jax.nn.softplus(-2.0 * entropy_raw))
        motor_ent = 4 * jnp.sum(log_std + 0.5 * math.log(2 * math.pi * math.e)) + log_jac.sum(axis=(-1, -2))
        cache = CommunicationState(up_cache, down_cache)
        policy = PolicyOutput(HybridAction(gu, pu, gd, pd, raw), jnp.tanh(raw), cache,
                              gate_lp + msg_lp + motor_lp, gate_ent + msg_ent + motor_ent,
                              gate_probs, gate_ent, msg_ent, motor_ent, gate_lp, msg_lp,
                              motor_lp, mean)
        new_state = MatchState(cache,
            _append(state.up_history, current_event, -3),
            _append(state.up_received, received_u, -2),
            _append(state.global_history, global_obs, -2),
            _append(state.local_history, local_obs, -3),
            _append(state.down_history, down_cache.payload, -3),
            _append(state.down_valid, down_cache.valid, -2),
            _append(state.down_age, down_cache.age, -2),
            _append(state.frame_valid, jnp.ones(batch, jnp.bool_), -1))
        return MatchOutput(policy, new_state, positions, attributes, attention)
