"""Equivalent sparse computations for the two-seed, four-GPU revision.

The original sparse_v1 module remains the frozen numerical reference. ASYM
retains exact enumeration for entropy, KL and the expected budget objective.
PERIODIC stores only its deterministic route's Gaussian and eight gate bits.
"""
import dataclasses
import math

from flax import linen as nn
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks

from .central_v1 import make_central_networks
from .full_v1 import SwishStack, transmit
from .sparse_v1 import SparseActor, HybridDistribution, periodic_flags, up_masks, select_branch


class FastSparseActor(SparseActor):
    @nn.compact
    def __call__(self, actor, *, up_gates=None, down_gates=None, fixed_down=None, fixed_up=None):
        aa, qq, vv = (jnp.asarray(x) for x in (self.actuator_indices, self.qpos_indices, self.qvel_indices))
        shape = actor.shape[:-1]
        global_obs = jnp.concatenate((actor[..., :9], actor[..., 45:48]), -1)
        local_obs = jnp.concatenate((actor[..., 9+qq], actor[..., 21+vv], actor[..., 33+aa]), -1)
        identity = jnp.broadcast_to(jnp.eye(4, dtype=actor.dtype), shape+(4, 4))
        local = SwishStack(self.local_widths, name='local_encoder')(jnp.concatenate((local_obs, identity), -1))
        up_payload = nn.Dense(self.message_dim, name='ascending', kernel_init=jax.nn.initializers.lecun_uniform())(local)
        small_init = lambda key, sh, dtype=jnp.float32: .01*jax.nn.initializers.lecun_uniform()(key, sh, dtype)
        bias_init = nn.initializers.constant(math.log(self.budget/(1-self.budget)))
        up_logit = nn.Dense(1, name='ascending_router', kernel_init=small_init, bias_init=bias_init)(local)[..., 0]
        global_module = SwishStack(self.global_widths, name='global_controller')
        down_module = nn.Dense(4*self.message_dim, name='descending', kernel_init=jax.nn.initializers.lecun_uniform())
        router_module = nn.Dense(1, name='descending_router', kernel_init=small_init, bias_init=bias_init)
        motor_module = SwishStack(self.motor_widths+(6,), activate_final=False, name='local_motor')
        if up_gates is None:
            delivered = up_payload[..., None, :, :]*up_masks()[..., None]
            glob = jnp.broadcast_to(global_obs[..., None, :], shape+(16, 12))
            hidden = global_module(jnp.concatenate((glob, delivered.reshape(shape+(16, 128))), -1))
            down = down_module(hidden).reshape(shape+(16, 4, 32))
            ident = jnp.broadcast_to(identity[..., None, :, :], shape+(16, 4, 4))
            hg = jnp.broadcast_to(hidden[..., None, :], shape+(16, 4, 128))
            down_logit = router_module(jnp.concatenate((hg, ident), -1))[..., 0]
            # A closed downlink has the same local input for every upper mask.
            off = motor_module(jnp.concatenate((local, jnp.zeros_like(up_payload)), -1))
            lh = jnp.broadcast_to(local[..., None, :, :], shape+(16, 4, 128))
            on = motor_module(jnp.concatenate((lh, down), -1))
            motor = jnp.stack((jnp.broadcast_to(off[..., None, :, :], on.shape), on), -2)
            return jnp.concatenate((up_logit, down_logit.reshape(shape+(64,)), motor.reshape(shape+(768,))), -1)
        up, up_wire = transmit(up_payload, up_gates.astype(bool))
        if fixed_up is not None:
            up = fixed_up
        hidden = global_module(jnp.concatenate((global_obs, up.reshape(shape+(128,))), -1))
        down_payload = down_module(hidden).reshape(shape+(4, 32))
        hg = jnp.broadcast_to(hidden[..., None, :], shape+(4, 128))
        down_logit = router_module(jnp.concatenate((hg, identity), -1))[..., 0]
        down, down_wire = transmit(down_payload, down_gates.astype(bool))
        if fixed_down is not None:
            down = fixed_down
        motor = motor_module(jnp.concatenate((local, down), -1))
        return motor, dict(ascending_payload=up, descending_payload=down, up_wire=up_wire, down_wire=down_wire,
                           ascending_router_logits=up_logit, descending_router_logits=down_logit)


class PeriodicGaussian:
    def __init__(self, parameters, actuator_indices):
        self.parameters = parameters
        self.loc = parameters[..., :12]
        self.scale = jax.nn.softplus(parameters[..., 12:24])+.001
        self.fixed = parameters[..., 24:32]
        self.aa = jnp.asarray(actuator_indices)

    def sample(self, seed, motor_noise=True):
        # Match the frozen three-key split and selected motor sampling exactly.
        _, _, key = jax.random.split(seed, 3)
        motor = self.loc+(self.scale*jax.random.normal(key, self.loc.shape) if motor_noise else 0)
        return jnp.concatenate((self.fixed, motor), -1)

    def mode(self):
        return jnp.concatenate((self.fixed, self.loc), -1)

    def log_prob(self, action):
        normal = jnp.sum(-.5*((action[..., 8:]-self.loc)/self.scale)**2-jnp.log(self.scale)-.5*jnp.log(2*jnp.pi), -1)
        return jnp.where(jnp.all(action[..., :8] == self.fixed, -1), normal, -jnp.inf)

    def entropy(self, seed):
        # Keep the original branchwise random stream: select its actual route.
        # Thus the MC entropy and its gradient retain the same estimator/key.
        shape = self.loc.shape[:-1]
        noise = jax.random.normal(seed, shape+(16, 4, 2, 3))
        noise = select_branch(noise, self.fixed[..., :4])
        noise = jnp.sum(noise*jax.nn.one_hot(self.fixed[..., 4:].astype(jnp.int32), 2)[..., None], -2)
        loc, scale = self.loc[..., self.aa], self.scale[..., self.aa]
        sample = loc+scale*noise
        jac = 2*(jnp.log(2.)-sample-jax.nn.softplus(-2*sample))
        return jnp.sum(.5*jnp.log(2*jnp.pi*jnp.e)+jnp.log(scale)+jac, axis=(-1, -2))

    def kl_divergence(self, old):
        value = jnp.sum(jnp.log(self.scale/old.scale)+(old.scale**2+(old.loc-self.loc)**2)/(2*self.scale**2)-.5, -1)
        return jnp.where(jnp.all(self.fixed == old.fixed, -1), value, jnp.inf)

    def expected_rates(self):
        return jnp.mean(self.fixed[..., :4], -1), jnp.mean(self.fixed[..., 4:], -1)


class CompactPeriodicDistribution(HybridDistribution):
    def __init__(self, actuator_indices):
        distribution.ParametricDistribution.__init__(self, 32, distribution.IdentityPostprocessor(), 1, False)
        self.actuator_indices = actuator_indices
        self.routing = 'periodic'

    def create_dist(self, parameters):
        return PeriodicGaussian(parameters, self.actuator_indices)


def make_sparse_components(observation_size, action_size, *, mapping, architecture, budget, routing, rates=None,
                           preprocess_observations_fn=types.identity_observation_preprocessor, **factory):
    if action_size != 20 or routing not in ('asym', 'periodic'):
        raise ValueError('Expected registered hybrid policy')
    central = make_central_networks(observation_size, 12, preprocess_observations_fn=preprocess_observations_fn, **factory)
    module = FastSparseActor(*(tuple(tuple(int(i) for i in row) for row in mapping[k])
                              for k in ('actuator_indices', 'qpos_relative_indices', 'qvel_relative_indices')),
                             budget=budget, local_widths=tuple(architecture['local_widths']),
                             global_widths=tuple(architecture['global_widths']), motor_widths=tuple(architecture['motor_widths']))
    rates = tuple(rates or (budget, budget))
    def actor_obs(normalizer, observation):
        return preprocess_observations_fn(observation['state'], networks.normalizer_select(normalizer, 'state'))
    def apply(normalizer, params, observation):
        flags = periodic_flags(observation['route_clock'][..., 0], rates)
        actor = actor_obs(normalizer, observation)
        if routing == 'asym':
            return jnp.concatenate((module.apply(params, actor), flags), -1)
        motor, _ = module.apply(params, actor, up_gates=flags[..., :4], down_gates=flags[..., 4:])
        aa = jnp.asarray(module.actuator_indices)
        loc = jnp.zeros(actor.shape[:-1]+(12,)).at[..., aa].set(motor[..., :3])
        raw_scale = jnp.zeros_like(loc).at[..., aa].set(motor[..., 3:])
        return jnp.concatenate((loc, raw_scale, flags), -1)
    def diagnostic(normalizer, params, observation, up, down, **kwargs):
        return module.apply(params, actor_obs(normalizer, observation), up_gates=up, down_gates=down, **kwargs)
    # Preserve the full named parameter tree, including PERIODIC's unused routers.
    policy = networks.FeedForwardNetwork(init=lambda rng: module.init(rng, jnp.zeros((1, 48))), apply=apply)
    value = networks.FeedForwardNetwork(init=central.value_network.init,
        apply=lambda n, p, o: central.value_network.apply(n, p, {k: o[k] for k in ('state', 'privileged_state')}))
    dist = HybridDistribution(routing, module.actuator_indices) if routing == 'asym' else CompactPeriodicDistribution(module.actuator_indices)
    return PPONetworks(policy, value, dist), diagnostic


def make_sparse_networks(observation_size, action_size, **kwargs):
    return make_sparse_components(observation_size, action_size, **kwargs)[0]


def budgeted_loss(original, coefficient, budget, routing):
    def loss(params, normalizer_params, data, rng, ppo_network, **kwargs):
        obs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data.observation)
        logits = ppo_network.policy_network.apply(normalizer_params, params.policy, obs)
        # The upstream loss's only policy call receives these exact observations.
        # Captured logits retain their gradient to params; nothing is detached.
        cached = dataclasses.replace(ppo_network.policy_network, apply=lambda *_: logits)
        network = ppo_network.replace(policy_network=cached)
        total, metrics = original(params, normalizer_params, data, rng, network, **kwargs)
        up, down = ppo_network.parametric_action_distribution.create_dist(logits).expected_rates()
        penalty = coefficient*jnp.mean((up-budget)**2+(down-budget)**2) if routing == 'asym' else jnp.asarray(0.)
        return total+penalty, {**metrics, 'total_loss': total+penalty, 'communication_budget_penalty': penalty,
                               'expected_up_fraction': up.mean(), 'expected_down_fraction': down.mean()}
    return loss
