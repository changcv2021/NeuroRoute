"""Always-delivered local/global actor on the accepted Central-V1 PPO contract.

Messages are deterministic float32 vectors. The measured wire format is a
logical interface, not physical network traffic: 32 bits per scalar plus one
send bit, two leg-address bits and one direction bit per transmitted packet.
No gates, budget penalty, prior, quantization or message sampling are learned.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from brax.training import networks, types
from brax.training.agents.ppo.networks import PPONetworks

from .central_v1 import central_observations, make_central_networks


def transmit(payload, send):
    """One packet per final-but-one axis; absent packets deliver/cost zero."""
    payload = jnp.asarray(payload, dtype=jnp.float32)
    send = jnp.asarray(send, dtype=jnp.bool_)
    if payload.shape[:-1] != send.shape:
        raise ValueError('Payload and send axes do not agree')
    delivered = jnp.where(send[..., None], payload, 0.)
    payload_bits = send.astype(jnp.int32) * (32 * payload.shape[-1])
    overhead_bits = send.astype(jnp.int32) * 4
    return delivered, {'send': send, 'payload_bits': payload_bits,
                       'overhead_bits': overhead_bits,
                       'total_bits': payload_bits + overhead_bits}


class SwishStack(nn.Module):
    widths: tuple
    activate_final: bool = True

    @nn.compact
    def __call__(self, x):
        for index, width in enumerate(self.widths):
            x = nn.Dense(width, name=f'hidden_{index}',
                         kernel_init=jax.nn.initializers.lecun_uniform())(x)
            if index < len(self.widths)-1 or self.activate_final:
                x = nn.swish(x)
        return x


class FullV1Actor(nn.Module):
    actuator_indices: tuple
    qpos_indices: tuple
    qvel_indices: tuple
    local_widths: tuple = (128, 128)
    global_widths: tuple = (512, 256, 128)
    motor_widths: tuple = (128, 64)
    message_dim: int = 32

    @nn.compact
    def __call__(self, actor, *, fixed_down=None):
        if actor.shape[-1] != 48:
            raise ValueError('FULL-V1 requires the accepted current 48D actor observation')
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
        up, up_wire = transmit(up_payload, jnp.ones(up_payload.shape[:-1], dtype=bool))
        global_hidden = SwishStack(self.global_widths, name='global_controller')(
            jnp.concatenate((global_obs, up.reshape(actor.shape[:-1]+(-1,))), axis=-1))
        down_payload = nn.Dense(4*self.message_dim, name='descending',
            kernel_init=jax.nn.initializers.lecun_uniform())(global_hidden)
        down_payload = down_payload.reshape(actor.shape[:-1]+(4, self.message_dim))
        down, down_wire = transmit(down_payload, jnp.ones(down_payload.shape[:-1], dtype=bool))
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
            'up_wire': up_wire, 'down_wire': down_wire}


def make_full_components(observation_size, action_size, *, mapping, architecture,
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
    module = FullV1Actor(
        *(tuple(tuple(int(i) for i in row) for row in mapping[key])
          for key in ('actuator_indices', 'qpos_relative_indices', 'qvel_relative_indices')),
        local_widths=tuple(architecture['local_widths']),
        global_widths=tuple(architecture['global_widths']),
        motor_widths=tuple(architecture['motor_widths']), message_dim=architecture['message_dim'])

    def diagnostic_apply(normalizer, params, observation, *, fixed_down=None):
        actor = central_observations(observation)['state']
        actor = preprocess_observations_fn(actor, networks.normalizer_select(normalizer, 'state'))
        return module.apply(params, actor, fixed_down=fixed_down)

    policy = networks.FeedForwardNetwork(
        init=lambda rng: module.init(rng, jnp.zeros((1, 48))),
        apply=lambda normalizer, params, observation: diagnostic_apply(normalizer, params, observation)[0])
    return PPONetworks(policy_network=policy, value_network=central.value_network,
                       parametric_action_distribution=central.parametric_action_distribution), diagnostic_apply


def make_full_networks(observation_size, action_size, **kwargs):
    return make_full_components(observation_size, action_size, **kwargs)[0]
