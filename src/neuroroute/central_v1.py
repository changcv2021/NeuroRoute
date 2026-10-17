"""Central V1 integration with the successful official Go1/PPO contract.

This is a new checkpoint/policy family. V0 implementations remain untouched.
The project FaultGo1 adapter is exercised with faults disabled. The PPO update
and normalization algorithms remain upstream; policy/value MLPs live here.
"""
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks

from .environment import FaultGo1
from .runtime import file_sha256, write_json


def central_observations(observation):
    if set(observation) != {'state', 'privileged_state'}:
        raise ValueError('Unexpected Central V1 observation keys')
    if observation['state'].shape[-1] != 48:
        raise ValueError('Central V1 requires current 48-dimensional state')
    # Keep the current official frame. No stacking, masking or actor privilege.
    return {key: observation[key] for key in ('state', 'privileged_state')}


class CentralV1Go1(FaultGo1):
    def __init__(self):
        super().__init__(task='flat_terrain', impl='jax', fault_kind='none', noise_level=1.0)

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(obs=central_observations(state.obs))

    def step(self, state, action):
        state = super().step(state, action)
        return state.replace(obs=central_observations(state.obs))


class CentralV1MLP(nn.Module):
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


def make_central_networks(observation_size, action_size,
                          preprocess_observations_fn=types.identity_observation_preprocessor,
                          policy_hidden_layer_sizes=(512, 256, 128),
                          value_hidden_layer_sizes=(512, 256, 128),
                          policy_obs_key='state', value_obs_key='privileged_state'):
    if policy_obs_key != 'state' or value_obs_key != 'privileged_state' or action_size != 12:
        raise ValueError('Central V1 observation/action contract changed')
    action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    def build(widths, key, scalar):
        module = CentralV1MLP(tuple(widths))
        shape = observation_size[key]
        size = int(shape if isinstance(shape, int) else shape[-1])
        dummy = jnp.zeros((1, size))

        def apply(normalizer, params, observation):
            observation = central_observations(observation)
            x = preprocess_observations_fn(observation[key], networks.normalizer_select(normalizer, key))
            output = module.apply(params, x)
            return jnp.squeeze(output, axis=-1) if scalar else output

        return networks.FeedForwardNetwork(init=lambda key: module.init(key, dummy), apply=apply)

    return PPONetworks(
        policy_network=build(tuple(policy_hidden_layer_sizes)+(action_distribution.param_size,), policy_obs_key, False),
        value_network=build(tuple(value_hidden_layer_sizes)+(1,), value_obs_key, True),
        parametric_action_distribution=action_distribution)


def save_checkpoint(path, params, contract, steps):
    """Trusted project-local pickle; bind preprocessing and exact source/config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {'format': 'neuroroute_central_v1', 'contract': contract,
               'environment_steps': int(steps), 'params': jax.device_get(params),
               'scope': 'normalizer_policy_value_not_complete_optimizer_resume'}
    with path.open('xb') as stream:
        pickle.dump(content, stream, protocol=pickle.HIGHEST_PROTOCOL)
    checksum = file_sha256(path)
    write_json(path.with_suffix('.json'), {'sha256': checksum, 'environment_steps': int(steps),
                                          'contract': contract, 'scope': content['scope']})
    return checksum


def load_checkpoint(path, checksum, contract):
    path = Path(path)
    if file_sha256(path) != checksum:
        raise ValueError('Central checkpoint hash mismatch')
    with path.open('rb') as stream:
        content = pickle.load(stream)
    if content['format'] != 'neuroroute_central_v1' or content['contract'] != contract:
        raise ValueError('Central checkpoint source/config/runtime contract mismatch')
    if file_sha256(path) != checksum:
        raise ValueError('Checkpoint changed during load')
    if not all(np.isfinite(x).all() for x in jax.tree.leaves(content['params'])):
        raise FloatingPointError('Nonfinite checkpoint parameters')
    # RunningStatisticsState.std_eps can be a Python float before pmap. Keep
    # scalar metadata unchanged; only stored device-array leaves go back to JAX.
    return jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
                        content['params'])
