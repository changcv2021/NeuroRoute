"""MATCH Q/TD paper-structure verification, not original-score reproduction.

Unlike the packet/PPO adapter, the hub recursively consumes its own previous
reconstructed states. TD replay runs each complete episode from a zero state
under the current parameters, and separately under the target parameters.
No cached behavior-network hidden states enter the learner.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn, struct


@dataclass(frozen=True)
class ReferenceConfig:
    num_agents: int = 5
    episode_length: int = 25
    width: int = 32
    heads: int = 4
    layers: int = 1
    history_length: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99
    eta_reconstruction: float = 0.1
    distance_scale: float = 1.0
    target_update_interval: int = 20
    max_grad_norm: float = 10.0
    mixer: str = "vdn_engineering_choice"
    mode: str = "match_reference_structure_verification"

    def validate(self):
        if self.num_agents not in (5, 6) or self.mode != "match_reference_structure_verification":
            raise ValueError("Reference checks only: original reproduction remains incomplete")
        if min(self.episode_length, self.width, self.heads, self.layers,
               self.history_length, self.target_update_interval) <= 0 or self.width % self.heads:
            raise ValueError("Invalid network/horizon configuration")
        if (self.learning_rate <= 0 or not 0 < self.gamma <= 1 or self.eta_reconstruction <= 0
                or self.distance_scale <= 0 or self.max_grad_norm <= 0
                or self.mixer != "vdn_engineering_choice"):
            raise ValueError("Invalid declared TD/reconstruction settings")


@struct.dataclass
class ReferenceState:
    local_hidden: jnp.ndarray
    reconstructed_history: jnp.ndarray
    history_valid: jnp.ndarray


@struct.dataclass
class ReferenceOutput:
    q_values: jnp.ndarray
    reconstruction: jnp.ndarray
    queries: jnp.ndarray
    replies: jnp.ndarray
    reply_weights: jnp.ndarray
    state: ReferenceState


@struct.dataclass
class ReferenceEpisode:
    entities: jnp.ndarray       # [T+1, ..., agents, entities, 6]
    visible: jnp.ndarray        # [T+1, ..., agents, entities]
    up_received: jnp.ndarray    # [T+1, ..., agents]
    down_received: jnp.ndarray
    targets: jnp.ndarray        # [T+1, ..., entities, 5], loss only
    actions: jnp.ndarray        # [T, ..., agents]
    rewards: jnp.ndarray        # [T, ...], shared environment reward once
    terminal: jnp.ndarray       # [T, ...], explicit finite-horizon boundary


@struct.dataclass
class ReferenceTrainState:
    params: object
    target_params: object
    optimizer: object
    updates: jnp.ndarray


def empty_reference_state(config, batch_shape=()):
    return ReferenceState(
        jnp.zeros(batch_shape + (config.num_agents, config.width), jnp.float32),
        jnp.zeros(batch_shape + (config.history_length, 2 * config.num_agents, 5), jnp.float32),
        jnp.zeros(batch_shape + (config.history_length,), jnp.bool_))


class ReferenceTransformer(nn.Module):
    width: int
    heads: int

    @nn.compact
    def __call__(self, x, valid):
        attention = nn.make_attention_mask(valid, valid, dtype=jnp.bool_)
        residual = nn.SelfAttention(num_heads=self.heads, qkv_features=self.width,
            out_features=self.width, dropout_rate=0.0)(nn.LayerNorm()(x),
                                                      mask=attention, deterministic=True)
        x = x + residual
        return x + nn.Dense(self.width)(nn.gelu(nn.Dense(2 * self.width)(nn.LayerNorm()(x))))


class ReferenceNetwork(nn.Module):
    config: ReferenceConfig

    @nn.compact
    def __call__(self, entities, visible, state, up_received, down_received):
        c = self.config
        count = 2 * c.num_agents
        # All actor inputs are agent-centric observations and their own carry.
        # True simulator targets deliberately do not appear in this signature.
        observations = jnp.where(visible[..., None], entities, 0.0)
        local = nn.Dense(c.width, name="local_embedding")(observations)
        local = jnp.concatenate((state.local_hidden[..., None, :], local), axis=-2)
        local_valid = jnp.concatenate((jnp.ones(visible.shape[:-1] + (1,), jnp.bool_), visible), axis=-1)
        for layer in range(c.layers):
            local = ReferenceTransformer(c.width, c.heads, name=f"local_transformer_{layer}")(local, local_valid)
        hidden = local[..., 0, :]
        queries = nn.Dense(5, name="query")(hidden)

        # Paper pooling is per ENTITY over all received agent perspectives.
        received_views = visible & up_received[..., None]
        sent_observations = jnp.where(received_views[..., None], observations, 0.0)
        values = nn.Dense(c.width, name="pool_projection")(sent_observations)
        scores = nn.Dense(1, use_bias=False, name="pool_attention")(values)[..., 0]
        weights = jax.nn.softmax(jnp.where(received_views, scores, -1e9), axis=-2)
        weights = jnp.where(received_views, weights, 0.0)
        pooled = jnp.sum(weights[..., None] * values, axis=-3)

        # Mandatory recursive mechanism: past PREDICTIONS, not cached input
        # packets, are projected and fed back into the next reconstruction.
        history = nn.Dense(c.width, name="history_projection")(state.reconstructed_history)
        tokens = jnp.concatenate((history, pooled[..., None, :, :]), axis=-3)
        temporal_position = self.param("temporal_position", nn.initializers.normal(.02),
                                       (c.history_length + 1, 1, c.width))
        entity_position = self.param("entity_position", nn.initializers.normal(.02), (1, count, c.width))
        tokens = tokens + temporal_position + entity_position
        batch = entities.shape[:-3]
        tokens = tokens.reshape(batch + ((c.history_length + 1) * count, c.width))
        frames = jnp.concatenate((state.history_valid,
                                  jnp.ones(batch + (1,), jnp.bool_)), axis=-1)
        valid = jnp.repeat(frames, count, axis=-1)
        for layer in range(c.layers):
            tokens = ReferenceTransformer(c.width, c.heads, name=f"hub_transformer_{layer}")(tokens, valid)
        reconstruction = nn.Dense(5, name="reconstruction")(tokens[..., -count:, :])

        received_queries = jnp.where(up_received[..., None], queries, 0.0)
        keys = nn.Dense(5, use_bias=False, name="answer_keys")(reconstruction)
        scores = jnp.einsum("...id,...jd->...ij", received_queries, keys) / jnp.sqrt(5.)
        relative = reconstruction[..., :c.num_agents, None, :2] - reconstruction[..., None, :, :2]
        # sqrt epsilon has zero derivative at coincident predicted positions.
        distance = jnp.sqrt(jnp.sum(relative ** 2, axis=-1) + 1e-12) / c.distance_scale
        locality = jax.nn.sigmoid(-distance)
        weight_scale = self.param("distance_attention_lambda", nn.initializers.ones, ())
        reply_weights = jax.nn.softmax(weight_scale * scores * locality, axis=-1)
        replies = jnp.einsum("...ij,...jd->...id", reply_weights, reconstruction)
        replies = jnp.where((up_received & down_received)[..., None], replies, 0.0)
        q_values = nn.Dense(5, name="q_values")(nn.gelu(nn.Dense(c.width, name="value_hidden")(
            jnp.concatenate((hidden, replies), axis=-1))))
        next_history = jnp.concatenate((state.reconstructed_history[..., 1:, :, :],
                                       reconstruction[..., None, :, :]), axis=-3)
        next_valid = jnp.concatenate((state.history_valid[..., 1:],
                                     jnp.ones(batch + (1,), jnp.bool_)), axis=-1)
        return ReferenceOutput(q_values, reconstruction, queries, replies, reply_weights,
                               ReferenceState(hidden, next_history, next_valid))


def reconstruct_episode(network, params, episode):
    """Differentiable full episode replay, starting at the actual reset."""
    batch = episode.entities.shape[1:-3]
    initial = empty_reference_state(network.config, batch)

    def step(carry, item):
        entities, visible, up, down = item
        output = network.apply({"params": params}, entities, visible, carry, up, down)
        return output.state, (output.q_values, output.reconstruction)

    return jax.lax.scan(step, initial, (episode.entities, episode.visible,
                                       episode.up_received, episode.down_received))[1]


def reference_auxiliary_loss(prediction, targets):
    truth = jax.lax.stop_gradient(targets)
    predicted_pos = prediction[..., :2] - prediction[..., :2].mean(axis=-2, keepdims=True)
    target_pos = truth[..., :2] - truth[..., :2].mean(axis=-2, keepdims=True)
    # Paper uses squared Euclidean norms then averages entities; time averaging
    # here is a declared scale convention, not a claimed published hyperparam.
    position = jnp.mean(jnp.sum((predicted_pos - target_pos) ** 2, axis=-1))
    attribute = jnp.mean(jnp.sum((prediction[..., 2:] - truth[..., 2:]) ** 2, axis=-1))
    return position, attribute


def td_objective(network, params, target_params, episode):
    q, predicted = reconstruct_episode(network, params, episode)
    target_q, _ = reconstruct_episode(network, target_params, episode)
    chosen = jnp.take_along_axis(q[:-1], episode.actions[..., None], axis=-1)[..., 0]
    # VDN is explicitly an engineering choice: MATCH does not specify its mixer.
    joint_q = chosen.sum(axis=-1)
    next_value = target_q[1:].max(axis=-1).sum(axis=-1)
    target = jax.lax.stop_gradient(episode.rewards + network.config.gamma *
                                  (1 - episode.terminal.astype(jnp.float32)) * next_value)
    td = jnp.mean((joint_q - target) ** 2)
    position, attribute = reference_auxiliary_loss(predicted, episode.targets)
    loss = td + network.config.eta_reconstruction * (position + attribute)
    return loss, {"loss": loss, "td_loss": td, "position_loss": position,
                  "attribute_loss": attribute, "joint_q_mean": joint_q.mean()}


class ReferenceLearner:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.network = ReferenceNetwork(config)
        self.optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                                     optax.adam(config.learning_rate))
        self.update = jax.jit(self._update)
        self.act = jax.jit(self._act)

    def initialize(self, seed):
        c = self.config
        params = self.network.init(jax.random.PRNGKey(seed), jnp.zeros((c.num_agents, 2*c.num_agents, 6)),
            jnp.ones((c.num_agents, 2*c.num_agents), jnp.bool_), empty_reference_state(c),
            jnp.ones((c.num_agents,), jnp.bool_), jnp.ones((c.num_agents,), jnp.bool_))["params"]
        return ReferenceTrainState(params, params, self.optimizer.init(params), jnp.array(0, jnp.int32))

    def _act(self, params, observation, visible, history, up, down, key, epsilon):
        output = self.network.apply({"params": params}, observation, visible, history, up, down)
        random_key, mask_key = jax.random.split(key)
        random_action = jax.random.randint(random_key, output.q_values.shape[:-1], 0, 5)
        random_mask = jax.random.uniform(mask_key, output.q_values.shape[:-1]) < epsilon
        action = jnp.where(random_mask, random_action, output.q_values.argmax(axis=-1))
        return action, output.state

    def _update(self, state, episode):
        objective = lambda params: td_objective(self.network, params, state.target_params, episode)
        (loss, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, optimizer = self.optimizer.update(gradient, state.optimizer, state.params)
        params = optax.apply_updates(state.params, updates)
        index = state.updates + jnp.array(1, jnp.int32)
        target = jax.tree.map(lambda new, old: jnp.where(index % self.config.target_update_interval == 0, new, old),
                              params, state.target_params)
        metrics = dict(metrics, gradient_norm=optax.global_norm(gradient))
        return ReferenceTrainState(params, target, optimizer, index), metrics
