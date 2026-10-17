"""FlyCNS-v2: causal Bernoulli communication with receiver memory.

Communication memory is deterministic actor state, not a stochastic action.
The PPO likelihood contains exactly eight gate decisions and twelve motors.
"""
import math
from typing import Callable

from flax import linen as nn, struct
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks

from .central_v1 import CentralV1Go1, make_central_networks
from .full_v1 import SwishStack, transmit
from .sparse_v1 import SparseActor, CausalDistribution, HybridDistribution, up_masks, select_branch
from .sparse_fast_v1 import make_sparse_components

CACHE_SIZE = 264


def initial_cache(shape=()):
    return jnp.concatenate((jnp.zeros(shape+(256,), jnp.float32),
                            jnp.full(shape+(8,), 1000., jnp.float32)), -1)


def unpack_cache(cache):
    shape = cache.shape[:-1]
    return (cache[..., :128].reshape(shape+(4, 32)),
            cache[..., 128:256].reshape(shape+(4, 32)),
            jnp.minimum(cache[..., 256:260]+1, 1000),
            jnp.minimum(cache[..., 260:264]+1, 1000))


def pack_cache(up, down, up_age, down_age):
    shape = up.shape[:-2]
    return jnp.concatenate((up.reshape(shape+(128,)), down.reshape(shape+(128,)), up_age, down_age), -1)


def curriculum_target(steps, budget, enabled=True):
    if not enabled:
        return jnp.asarray(budget, jnp.float32)
    values = (1., .9, .8, .75) if budget == .75 else (1., .8, .65, .5)
    stage = jnp.sum(jnp.asarray(steps) >= jnp.asarray((45875200, 91750400, 137625600)))
    return jnp.asarray(values, jnp.float32)[stage]


class FlyCNSGo1(CentralV1Go1):
    def __init__(self, budget=.75, **kwargs):
        self.final_budget = budget
        super().__init__(**kwargs)

    @property
    def action_size(self):
        return 20

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(obs={**state.obs, 'communication': initial_cache(),
            'route_clock': jnp.zeros((1,), jnp.float32),
            'routing_context': jnp.asarray((self.final_budget, 0.), jnp.float32)})

    def step(self, state, action):
        if action.shape[-1] != 20:
            raise ValueError('Expected eight gates and twelve motors')
        memory, clock, context = (state.obs[k] for k in ('communication', 'route_clock', 'routing_context'))
        physical = state.replace(obs={k: state.obs[k] for k in ('state', 'privileged_state')})
        nxt = super().step(physical, action[8:])
        return nxt.replace(obs={**nxt.obs, 'communication': memory,
                                'route_clock': clock+1, 'routing_context': context})


class CachedActor(SparseActor):
    use_age: bool = True

    @nn.compact
    def __call__(self, actor, cache, *, up_gates=None, down_gates=None, fixed_down=None, fixed_up=None):
        aa, qq, vv = (jnp.asarray(x) for x in (self.actuator_indices, self.qpos_indices, self.qvel_indices))
        shape = actor.shape[:-1]
        up_old, down_old, up_age, down_age = unpack_cache(cache)
        age_feature = lambda x: jnp.log1p(x)/jnp.log(1001.) if self.use_age else jnp.zeros_like(x)
        global_obs = jnp.concatenate((actor[..., :9], actor[..., 45:48]), -1)
        local_obs = jnp.concatenate((actor[..., 9+qq], actor[..., 21+vv], actor[..., 33+aa]), -1)
        identity = jnp.broadcast_to(jnp.eye(4, dtype=actor.dtype), shape+(4, 4))
        local = SwishStack(self.local_widths, name='local_encoder')(jnp.concatenate((local_obs, identity), -1))
        up_payload = nn.Dense(32, name='ascending', kernel_init=jax.nn.initializers.lecun_uniform())(local)
        small_init = lambda key, sh, dtype=jnp.float32: .01*jax.nn.initializers.lecun_uniform()(key, sh, dtype)
        bias_init = nn.initializers.constant(math.log(self.budget/(1-self.budget)))
        up_logit = nn.Dense(1, name='ascending_router', kernel_init=small_init, bias_init=bias_init)(local)[..., 0]
        global_module = SwishStack(self.global_widths, name='global_controller')
        down_module = nn.Dense(128, name='descending', kernel_init=jax.nn.initializers.lecun_uniform())
        router_module = nn.Dense(1, name='descending_router', kernel_init=small_init, bias_init=bias_init)
        motor_module = SwishStack(self.motor_widths+(6,), activate_final=False, name='local_motor')
        if up_gates is None:
            masks = up_masks().astype(bool)
            delivered = jnp.where(masks[..., None], up_payload[..., None, :, :], up_old[..., None, :, :])
            ages = jnp.where(masks, 0., up_age[..., None, :])
            glob = jnp.broadcast_to(global_obs[..., None, :], shape+(16, 12))
            hidden = global_module(jnp.concatenate((glob, delivered.reshape(shape+(16, 128)), age_feature(ages)), -1))
            down = down_module(hidden).reshape(shape+(16, 4, 32))
            ident = jnp.broadcast_to(identity[..., None, :, :], shape+(16, 4, 4))
            hg = jnp.broadcast_to(hidden[..., None, :], shape+(16, 4, 128))
            down_logit = router_module(jnp.concatenate((hg, ident), -1))[..., 0]
            off = motor_module(jnp.concatenate((local, down_old, age_feature(down_age)[..., None]), -1))
            lh = jnp.broadcast_to(local[..., None, :, :], shape+(16, 4, 128))
            on = motor_module(jnp.concatenate((lh, down, jnp.zeros(shape+(16, 4, 1))), -1))
            motor = jnp.stack((jnp.broadcast_to(off[..., None, :, :], on.shape), on), -2)
            return jnp.concatenate((up_logit, down_logit.reshape(shape+(64,)), motor.reshape(shape+(768,))), -1)
        up_wire_payload, up_wire = transmit(up_payload, up_gates.astype(bool))
        up = jnp.where(up_gates[..., None].astype(bool), up_payload, up_old)
        ua = jnp.where(up_gates.astype(bool), 0., up_age)
        if fixed_up is not None:
            up = fixed_up
        hidden = global_module(jnp.concatenate((global_obs, up.reshape(shape+(128,)), age_feature(ua)), -1))
        down_payload = down_module(hidden).reshape(shape+(4, 32))
        hg = jnp.broadcast_to(hidden[..., None, :], shape+(4, 128))
        down_logit = router_module(jnp.concatenate((hg, identity), -1))[..., 0]
        down_wire_payload, down_wire = transmit(down_payload, down_gates.astype(bool))
        down = jnp.where(down_gates[..., None].astype(bool), down_payload, down_old)
        da = jnp.where(down_gates.astype(bool), 0., down_age)
        if fixed_down is not None:
            down = fixed_down
        motor = motor_module(jnp.concatenate((local, down, age_feature(da)[..., None]), -1))
        return motor, dict(ascending_payload=up_wire_payload, descending_payload=down_wire_payload,
            up_wire=up_wire, down_wire=down_wire, ascending_router_logits=up_logit,
            descending_router_logits=down_logit, received_up=up, received_down=down,
            up_age=ua, down_age=da, next_communication_state=pack_cache(up, down, ua, da))


def probability_log_prob(probability, value):
    # Exact zero/one support, with finite internal logs and zero saturated gradients.
    yes = jnp.log(jnp.maximum(probability, jnp.finfo(jnp.float32).tiny))
    no = jnp.log(jnp.maximum(1-probability, jnp.finfo(jnp.float32).tiny))
    return value*yes+(1-value)*no


class CurriculumDistribution(CausalDistribution):
    def __init__(self, parameters, actuator_indices, budget):
        super().__init__(parameters[..., :844], 'asym', actuator_indices)
        self.parameters = parameters
        target = parameters[..., 844]
        floor = jnp.clip((target-budget)/(1-budget), 0., 1.)
        self.up_prob = floor[..., None]+(1-floor[..., None])*jax.nn.sigmoid(self.up)
        self.down_prob = floor[..., None, None]+(1-floor[..., None, None])*jax.nn.sigmoid(self.down)
        self.up_weights = jnp.prod(jnp.where(up_masks()==1, self.up_prob[..., None, :], 1-self.up_prob[..., None, :]), -1)
        self.down_weights = jnp.stack((1-self.down_prob, self.down_prob), -1)

    def log_prob(self, action):
        up, down, motor = action[..., :4], action[..., 4:8], action[..., 8:]
        loc, scale = self.selected_motor(up, down)
        normal = jnp.sum(-.5*((motor-loc)/scale)**2-jnp.log(scale)-.5*jnp.log(2*jnp.pi), -1)
        dp = select_branch(self.down_prob, up)
        gate = jnp.sum(probability_log_prob(self.up_prob, up)+probability_log_prob(dp, down), -1)
        invalid = jnp.any(((self.up_prob==1)&(up==0))|((self.up_prob==0)&(up==1))|
                          ((dp==1)&(down==0))|((dp==0)&(down==1)), -1)
        return normal+jnp.where(invalid, -jnp.inf, gate)

    def entropy(self, seed):
        sample = self.loc+self.scale*jax.random.normal(seed, self.loc.shape)
        jac = 2*(jnp.log(2.)-sample-jax.nn.softplus(-2*sample))
        motor = jnp.sum(.5*jnp.log(2*jnp.pi*jnp.e)+jnp.log(self.scale)+jac, -1)
        hu = -probability_log_prob(self.up_prob, self.up_prob)
        hd = -probability_log_prob(self.down_prob, self.down_prob)
        return self.expectation(motor)+jnp.sum(hu, -1)+jnp.sum(self.up_weights*jnp.sum(hd, -1), -1)

    def kl_divergence(self, old):
        motor = jnp.sum(jnp.log(self.scale/old.scale)+(old.scale**2+(old.loc-self.loc)**2)/(2*self.scale**2)-.5, -1)
        ku = probability_log_prob(old.up_prob, old.up_prob)-probability_log_prob(self.up_prob, old.up_prob)
        kd = probability_log_prob(old.down_prob, old.down_prob)-probability_log_prob(self.down_prob, old.down_prob)
        value = old.expectation(motor)+jnp.sum(ku, -1)+jnp.sum(old.up_weights*jnp.sum(kd, -1), -1)
        invalid_up = jnp.any(((self.up_prob==1)&(old.up_prob<1))|((self.up_prob==0)&(old.up_prob>0)), -1)
        invalid_down = jnp.any(((self.down_prob==1)&(old.down_prob<1))|((self.down_prob==0)&(old.down_prob>0)), -1)
        invalid = invalid_up|jnp.any(invalid_down & (old.up_weights>0), -1)
        return jnp.where(invalid, jnp.inf, value)


class CurriculumHybrid(HybridDistribution):
    def __init__(self, actuator_indices, budget):
        distribution.ParametricDistribution.__init__(self, 845, distribution.IdentityPostprocessor(), 1, False)
        self.actuator_indices, self.budget, self.routing = actuator_indices, budget, 'asym'

    def create_dist(self, parameters):
        return CurriculumDistribution(parameters, self.actuator_indices, self.budget)


@struct.dataclass
class MemoryNetworks(PPONetworks):
    advance: Callable = struct.field(pytree_node=False)
    replay: Callable = struct.field(pytree_node=False)
    curriculum: bool = struct.field(pytree_node=False)
    adaptive_dual: bool = struct.field(pytree_node=False)
    budget: float = struct.field(pytree_node=False)
    bio_coefficient: float = struct.field(pytree_node=False)
    pi_bio: tuple = struct.field(pytree_node=False)


def make_components(observation_size, action_size, *, mapping, architecture, budget, variant, pi_bio,
                    bio_coefficient=.01, preprocess_observations_fn=types.identity_observation_preprocessor, **factory):
    if action_size != 20:
        raise ValueError('Expected hybrid action size20')
    if variant['cache']:
        module = CachedActor(*(tuple(tuple(int(i) for i in row) for row in mapping[k])
            for k in ('actuator_indices', 'qpos_relative_indices', 'qvel_relative_indices')),
            budget=budget, local_widths=tuple(architecture['local_widths']),
            global_widths=tuple(architecture['global_widths']), motor_widths=tuple(architecture['motor_widths']),
            use_age=variant['age'])
        central = make_central_networks(observation_size, 12, preprocess_observations_fn=preprocess_observations_fn, **factory)
        def actor_obs(n, o):
            return preprocess_observations_fn(o['state'], networks.normalizer_select(n, 'state'))
        def apply(n, p, o):
            table = module.apply(p, actor_obs(n, o), o['communication'])
            return jnp.concatenate((table, jnp.zeros(table.shape[:-1]+(8,)), o['routing_context'][..., :1]), -1)
        def diagnostic(n, p, o, up, down, **kwargs):
            return module.apply(p, actor_obs(n, o), o['communication'], up_gates=up, down_gates=down, **kwargs)
        policy = networks.FeedForwardNetwork(init=lambda rng: module.init(rng, jnp.zeros((1, 48)), initial_cache((1,))), apply=apply)
        value = networks.FeedForwardNetwork(init=central.value_network.init,
            apply=lambda n, p, o: central.value_network.apply(n, p, {k:o[k] for k in ('state', 'privileged_state')}))
        dist = CurriculumHybrid(module.actuator_indices, budget)
        def replay(n, p, observations, gates):
            def step(cache, item):
                obs, gate = item
                cache = jnp.where(obs['route_clock']==0, initial_cache(cache.shape[:-1]), cache)
                obs = {**obs, 'communication': cache}
                logits = apply(n, p, obs)
                _, route = diagnostic(n, p, obs, gate[..., :4], gate[..., 4:8])
                return route['next_communication_state'], logits
            cache = jax.lax.stop_gradient(observations['communication'][0])
            return jax.lax.scan(jax.checkpoint(step), cache, (observations, gates))[1]
    else:
        # BIO-v1 uses the frozen actor verbatim; no cache, age or curriculum path.
        old, old_diagnostic = make_sparse_components(observation_size, action_size, mapping=mapping,
            architecture=architecture, budget=budget, routing='asym',
            preprocess_observations_fn=preprocess_observations_fn, **factory)
        policy, value, dist = old.policy_network, old.value_network, old.parametric_action_distribution
        def diagnostic(n, p, o, up, down, **kwargs):
            motor, route = old_diagnostic(n, p, o, up, down, **kwargs)
            return motor, {**route, 'next_communication_state': initial_cache(o['state'].shape[:-1])}
        def replay(n, p, observations, gates):
            return policy.apply(n, p, observations)
    network = MemoryNetworks(policy, value, dist, diagnostic, replay, variant['curriculum'],
        variant['adaptive_dual'], budget, bio_coefficient if variant['bio'] else 0., tuple(pi_bio))
    return network, diagnostic


def make_networks(observation_size, action_size, **kwargs):
    return make_components(observation_size, action_size, **kwargs)[0]


def make_inference_fn(network, compute_value=False, use_distributional_critic=False):
    if use_distributional_critic:
        raise ValueError('Registered scalar critic required')
    def make_policy(params, deterministic=False):
        def policy(obs, key):
            logits = network.policy_network.apply(params[0], params[1], obs)
            dist = network.parametric_action_distribution
            raw = dist.create_dist(logits).mode() if deterministic else dist.sample_no_postprocessing(logits, key)
            _, route = network.advance(params[0], params[1], obs, raw[..., :4], raw[..., 4:8])
            extras = {'raw_action':raw, 'log_prob':dist.log_prob(logits, raw), 'distribution_params':logits,
                      'next_communication_state':route['next_communication_state']}
            if compute_value:
                extras['value'] = network.value_network.apply(params[0], params[2], obs)
            return dist.postprocess(raw), extras
        return policy
    return make_policy
