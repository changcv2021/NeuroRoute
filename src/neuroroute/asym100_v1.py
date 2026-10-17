"""Phase4A: separate ascending/descending routing heads, gates fixed open.

This control preserves the FULL computation and packet format. Gate scores
are recorded but intentionally have no action effect or PPO gradient here.
Sparse routing requires a separately validated Phase4B sampler/objective.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from brax.training import networks, types
from brax.training.agents.ppo.networks import PPONetworks

from .central_v1 import central_observations, make_central_networks


from .full_v1 import transmit, SwishStack


def open_gates(logits, *, mode='forced_open'):
    if mode != 'forced_open':
        raise ValueError('Sparse gate execution is not validated by Phase4A')
    return jnp.ones(logits.shape,dtype=bool)


class Asym100V1Actor(nn.Module):
    actuator_indices: tuple
    qpos_indices: tuple
    qvel_indices: tuple
    local_widths: tuple = (128, 128)
    global_widths: tuple = (512, 256, 128)
    motor_widths: tuple = (128, 64)
    message_dim: int = 32

    @nn.compact
    def __call__(self, actor, *, fixed_down=None, fixed_up=None):
        if actor.shape[-1] != 48:
            raise ValueError('ASYM100-V1 requires the accepted current 48D actor observation')
        aa, qq, vv = (jnp.asarray(x) for x in
                      (self.actuator_indices, self.qpos_indices, self.qvel_indices))
        global_obs = jnp.concatenate((actor[..., :9], actor[..., 45:48]), axis=-1)
        local_obs = jnp.concatenate((actor[..., 9+qq], actor[..., 21+vv],
                                     actor[..., 33+aa]), axis=-1)
        identity = jnp.broadcast_to(jnp.eye(4, dtype=actor.dtype), actor.shape[:-1]+(4, 4))
        local_hidden = SwishStack(self.local_widths, name='local_encoder')(
            jnp.concatenate((local_obs, identity), axis=-1))
        up_payload = nn.Dense(self.message_dim, name='ascending',
            kernel_init=jax.nn.initializers.lecun_uniform())(local_hidden)
        up_logits = nn.Dense(1, name='ascending_router',
            kernel_init=jax.nn.initializers.lecun_uniform())(local_hidden)[..., 0]
        up_send = open_gates(up_logits)
        up, up_wire = transmit(up_payload, up_send)
        if fixed_up is not None:
            if fixed_up.shape != up.shape:
                raise ValueError('Fixed ascending packet shape mismatch')
            up = fixed_up
        global_hidden = SwishStack(self.global_widths, name='global_controller')(
            jnp.concatenate((global_obs, up.reshape(actor.shape[:-1]+(-1,))), axis=-1))
        down_payload = nn.Dense(4*self.message_dim, name='descending',
            kernel_init=jax.nn.initializers.lecun_uniform())(global_hidden)
        down_payload = down_payload.reshape(actor.shape[:-1]+(4, self.message_dim))
        per_leg_global = jnp.broadcast_to(global_hidden[..., None, :],
            actor.shape[:-1]+(4, self.global_widths[-1]))
        down_logits = nn.Dense(1, name='descending_router',
            kernel_init=jax.nn.initializers.lecun_uniform())(
                jnp.concatenate((per_leg_global, identity), axis=-1))[..., 0]
        down_send = open_gates(down_logits)
        down, down_wire = transmit(down_payload, down_send)
        # Audit intervention only. It does not add a motor path around packets.
        if fixed_down is not None:
            if fixed_down.shape != down.shape:
                raise ValueError('Fixed descending packet shape mismatch')
            down = fixed_down
        local_logits = SwishStack(self.motor_widths+(6,), activate_final=False,
                                 name='local_motor')(
            jnp.concatenate((local_hidden, down), axis=-1))
        # Brax NormalTanhDistribution expects all 12 locations then 12 scales.
        loc = jnp.zeros(actor.shape[:-1]+(12,), actor.dtype).at[..., aa].set(local_logits[..., :3])
        scale = jnp.zeros_like(loc).at[..., aa].set(local_logits[..., 3:])
        logits = jnp.concatenate((loc, scale), axis=-1)
        return logits, {'global_observation': global_obs, 'local_observation': local_obs,
            'local_hidden': local_hidden, 'ascending_payload': up,
            'descending_payload': down, 'local_logits': local_logits,
            'up_wire': up_wire, 'down_wire': down_wire,
            'ascending_router_logits': up_logits, 'descending_router_logits': down_logits}


def make_asym100_components(observation_size, action_size, *, mapping, architecture,
                         preprocess_observations_fn=types.identity_observation_preprocessor,
                         policy_hidden_layer_sizes=(512, 256, 128),
                         value_hidden_layer_sizes=(512, 256, 128),
                         policy_obs_key='state', value_obs_key='privileged_state'):
    if tuple(policy_hidden_layer_sizes) != tuple(architecture['global_widths']):
        raise ValueError('Global hidden dimensions must match accepted Central hidden dimensions')
    central = make_central_networks(observation_size, action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        policy_obs_key=policy_obs_key, value_obs_key=value_obs_key)
    module = Asym100V1Actor(
        *(tuple(tuple(int(i) for i in row) for row in mapping[key])
          for key in ('actuator_indices', 'qpos_relative_indices', 'qvel_relative_indices')),
        local_widths=tuple(architecture['local_widths']),
        global_widths=tuple(architecture['global_widths']),
        motor_widths=tuple(architecture['motor_widths']), message_dim=architecture['message_dim'])

    def diagnostic_apply(normalizer, params, observation, *, fixed_down=None, fixed_up=None):
        actor = central_observations(observation)['state']
        actor = preprocess_observations_fn(actor, networks.normalizer_select(normalizer, 'state'))
        return module.apply(params, actor, fixed_down=fixed_down, fixed_up=fixed_up)

    policy = networks.FeedForwardNetwork(
        init=lambda rng: module.init(rng, jnp.zeros((1, 48))),
        apply=lambda normalizer, params, observation: diagnostic_apply(normalizer, params, observation)[0])
    return PPONetworks(policy_network=policy, value_network=central.value_network,
                       parametric_action_distribution=central.parametric_action_distribution), diagnostic_apply


def make_asym100_networks(observation_size, action_size, **kwargs):
    return make_asym100_components(observation_size, action_size, **kwargs)[0]
