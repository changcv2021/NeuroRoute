"""Packet transport, shared MLP layers, and the reference value network."""
import jax
import jax.numpy as jnp
from flax import linen as nn
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks
from ..environment import current_observations

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

class ReferenceMLP(nn.Module):
    widths: tuple

    @nn.compact
    def __call__(self, observation):
        x = observation
        for index, width in enumerate(self.widths):
            x = nn.Dense(width, name=f'hidden_{index}',
                         kernel_init=jax.nn.initializers.lecun_uniform())(x)
            if index < len(self.widths) - 1:
                x = nn.swish(x)
        return x

def make_reference_networks(observation_size, action_size,
                          preprocess_observations_fn=types.identity_observation_preprocessor,
                          policy_hidden_layer_sizes=(512, 256, 128),
                          value_hidden_layer_sizes=(512, 256, 128),
                          policy_obs_key='state', value_obs_key='privileged_state'):
    if policy_obs_key != 'state' or value_obs_key != 'privileged_state' or action_size != 12:
        raise ValueError('Go1 observation/action contract changed')
    action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    def build(widths, key, scalar):
        module = ReferenceMLP(tuple(widths))
        shape = observation_size[key]
        size = int(shape if isinstance(shape, int) else shape[-1])
        dummy = jnp.zeros((1, size))

        def apply(normalizer, params, observation):
            observation = current_observations(observation)
            x = preprocess_observations_fn(observation[key], networks.normalizer_select(normalizer, key))
            output = module.apply(params, x)
            return jnp.squeeze(output, axis=-1) if scalar else output

        return networks.FeedForwardNetwork(init=lambda key: module.init(key, dummy), apply=apply)

    return PPONetworks(
        policy_network=build(tuple(policy_hidden_layer_sizes)+(action_distribution.param_size,), policy_obs_key, False),
        value_network=build(tuple(value_hidden_layer_sizes)+(1,), value_obs_key, True),
        parametric_action_distribution=action_distribution)
