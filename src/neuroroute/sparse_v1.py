"""Gate4B causal Bernoulli communication and conditional Gaussian motors.

Counterfactual branches are a probability representation, not delivered packets.
The environment receives eight exact gate bits and twelve motor actions.
"""
import math

from flax import linen as nn
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks

from .central_v1 import CentralV1Go1, make_central_networks
from .full_v1 import SwishStack, transmit


def up_masks():
    return ((jnp.arange(16)[:, None] >> jnp.arange(4)) & 1).astype(jnp.float32)


def select_branch(values, gates):
    """Select the 16-way axis immediately after the batch axes."""
    index = jnp.sum(gates.astype(jnp.int32) * (1 << jnp.arange(4)), axis=-1)
    extra = values.ndim - gates.ndim
    selection = index.reshape(index.shape + (1,) * (extra + 1))
    return jnp.take_along_axis(values, selection, axis=index.ndim).squeeze(index.ndim)


def periodic_flags(step, rates):
    """Public accumulator schedule, integer millionth rates, fixed leg phases."""
    numerator = jnp.asarray([round(float(r) * 1_000_000) for r in rates], jnp.int32)
    # step is reduced first, so int32 remains exact within an episode (<1000).
    phase = jnp.asarray([0, 250000, 500000, 750000], jnp.int32)
    position = jnp.mod(step.astype(jnp.int32)[..., None, None] * numerator[:, None] + phase, 1_000_000)
    return (position + numerator[:, None] >= 1_000_000).astype(jnp.float32).reshape(step.shape+(8,))


class SparseV1Go1(CentralV1Go1):
    @property
    def action_size(self):
        return 20

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info); info['route_step'] = jnp.int32(0)
        obs = dict(state.obs); obs['route_clock'] = jnp.zeros((1,), jnp.float32)
        return state.replace(info=info, obs=obs)

    def step(self, state, action):
        if action.shape[-1] != 20:
            raise ValueError('Sparse action must contain 8 gates and 12 motors')
        clock = state.obs['route_clock'][0]
        # Strip the public scheduling observation before the strict Central adapter.
        state = state.replace(obs={k:state.obs[k] for k in ('state','privileged_state')})
        nxt = super().step(state, action[8:])
        info = dict(nxt.info); info['route_step'] = clock.astype(jnp.int32)+1
        obs = dict(nxt.obs); obs['route_clock'] = jnp.asarray([clock+1], jnp.float32)
        return nxt.replace(info=info, obs=obs)


class SparseActor(nn.Module):
    actuator_indices: tuple
    qpos_indices: tuple
    qvel_indices: tuple
    budget: float
    local_widths: tuple = (128,128)
    global_widths: tuple = (512,256,128)
    motor_widths: tuple = (128,64)
    message_dim: int = 32

    @nn.compact
    def __call__(self, actor, *, up_gates=None, down_gates=None, fixed_down=None, fixed_up=None):
        aa,qq,vv = (jnp.asarray(x) for x in (self.actuator_indices,self.qpos_indices,self.qvel_indices))
        shape=actor.shape[:-1]
        global_obs=jnp.concatenate((actor[...,:9],actor[...,45:48]),-1)
        local_obs=jnp.concatenate((actor[...,9+qq],actor[...,21+vv],actor[...,33+aa]),-1)
        identity=jnp.broadcast_to(jnp.eye(4,dtype=actor.dtype),shape+(4,4))
        local=SwishStack(self.local_widths,name='local_encoder')(jnp.concatenate((local_obs,identity),-1))
        up_payload=nn.Dense(self.message_dim,name='ascending',kernel_init=jax.nn.initializers.lecun_uniform())(local)
        small_init=lambda key,sh,dtype=jnp.float32: .01*jax.nn.initializers.lecun_uniform()(key,sh,dtype)
        bias_init=nn.initializers.constant(math.log(self.budget/(1-self.budget)))
        up_logit=nn.Dense(1,name='ascending_router',kernel_init=small_init,bias_init=bias_init)(local)[...,0]
        global_module=SwishStack(self.global_widths,name='global_controller')
        down_module=nn.Dense(4*self.message_dim,name='descending',kernel_init=jax.nn.initializers.lecun_uniform())
        router_module=nn.Dense(1,name='descending_router',kernel_init=small_init,bias_init=bias_init)
        motor_module=SwishStack(self.motor_widths+(6,),activate_final=False,name='local_motor')
        if up_gates is None:
            delivered=up_payload[...,None,:,:]*up_masks()[...,None]
            glob=jnp.broadcast_to(global_obs[...,None,:],shape+(16,12))
            hidden=global_module(jnp.concatenate((glob,delivered.reshape(shape+(16,128))),-1))
            down=down_module(hidden).reshape(shape+(16,4,32))
            ident=jnp.broadcast_to(identity[...,None,:,:],shape+(16,4,4))
            hg=jnp.broadcast_to(hidden[...,None,:],shape+(16,4,128))
            down_logit=router_module(jnp.concatenate((hg,ident),-1))[...,0]
            lh=jnp.broadcast_to(local[...,None,:,None,:],shape+(16,4,2,128))
            dh=down[...,None,:]*jnp.asarray([0.,1.])[:,None]
            motor=motor_module(jnp.concatenate((lh,dh),-1))
            return jnp.concatenate((up_logit,down_logit.reshape(shape+(64,)),motor.reshape(shape+(768,))),-1)
        up,up_wire=transmit(up_payload,up_gates.astype(bool))
        if fixed_up is not None: up=fixed_up
        hidden=global_module(jnp.concatenate((global_obs,up.reshape(shape+(128,))),-1))
        down_payload=down_module(hidden).reshape(shape+(4,32))
        hg=jnp.broadcast_to(hidden[...,None,:],shape+(4,128))
        down_logit=router_module(jnp.concatenate((hg,identity),-1))[...,0]
        down,down_wire=transmit(down_payload,down_gates.astype(bool))
        if fixed_down is not None: down=fixed_down
        motor=motor_module(jnp.concatenate((local,down),-1))
        return motor,dict(ascending_payload=up,descending_payload=down,up_wire=up_wire,down_wire=down_wire,
                         ascending_router_logits=up_logit,descending_router_logits=down_logit)


def bernoulli_log_prob(logits, values):
    return values*jax.nn.log_sigmoid(logits)+(1-values)*jax.nn.log_sigmoid(-logits)


class CausalDistribution:
    def __init__(self, parameters, routing, actuator_indices):
        self.parameters=parameters; self.routing=routing; self.aa=jnp.asarray(actuator_indices)
        shape=parameters.shape[:-1]
        self.up=parameters[...,:4]
        self.down=parameters[...,4:68].reshape(shape+(16,4))
        motor=parameters[...,68:836].reshape(shape+(16,4,2,6))
        self.loc=motor[...,:3]; self.scale=jax.nn.softplus(motor[...,3:])+.001
        self.fixed=parameters[...,836:844]
        if routing=='asym':
            self.up_prob=jax.nn.sigmoid(self.up); self.down_prob=jax.nn.sigmoid(self.down)
            self.up_weights=jnp.exp(jnp.sum(bernoulli_log_prob(self.up[...,None,:],up_masks()),-1))
        else:
            self.up_prob=self.fixed[...,:4]
            self.down_prob=jnp.broadcast_to(self.fixed[...,None,4:],shape+(16,4))
            self.up_weights=jnp.prod(jnp.where(up_masks()==1,self.up_prob[...,None,:],1-self.up_prob[...,None,:]),-1)
        self.down_weights=jnp.stack((1-self.down_prob,self.down_prob),-1)

    def selected_motor(self, up, down):
        loc=select_branch(self.loc,up); scale=select_branch(self.scale,up)
        loc=jnp.sum(loc*jax.nn.one_hot(down.astype(jnp.int32),2)[...,None],-2)
        scale=jnp.sum(scale*jax.nn.one_hot(down.astype(jnp.int32),2)[...,None],-2)
        global_loc=jnp.zeros(up.shape[:-1]+(12,)).at[...,self.aa].set(loc)
        global_scale=jnp.zeros_like(global_loc).at[...,self.aa].set(scale)
        return global_loc,global_scale

    def sample(self, seed, motor_noise=True):
        k1,k2,k3=jax.random.split(seed,3)
        up=jax.random.bernoulli(k1,self.up_prob).astype(jnp.float32)
        down=jax.random.bernoulli(k2,select_branch(self.down_prob,up)).astype(jnp.float32)
        loc,scale=self.selected_motor(up,down)
        motor=loc+(jax.random.normal(k3,loc.shape)*scale if motor_noise else 0)
        return jnp.concatenate((up,down,motor),-1)

    def mode(self):
        # Used only by upstream diagnostic eval, not the registered gate evaluation.
        up=(self.up_prob>=.5).astype(jnp.float32)
        down=(select_branch(self.down_prob,up)>=.5).astype(jnp.float32)
        return jnp.concatenate((up,down,self.selected_motor(up,down)[0]),-1)

    def log_prob(self, action):
        up,down,motor=action[...,:4],action[...,4:8],action[...,8:]
        loc,scale=self.selected_motor(up,down)
        normal=jnp.sum(-.5*jnp.square((motor-loc)/scale)-jnp.log(scale)-.5*jnp.log(2*jnp.pi),-1)
        if self.routing=='asym':
            gate=jnp.sum(bernoulli_log_prob(self.up,up)+bernoulli_log_prob(select_branch(self.down,up),down),-1)
        else:
            gate=jnp.where(jnp.all(action[...,:8]==self.fixed,-1),0.,-jnp.inf)
        return gate+normal

    def expectation(self, per_leg_state):
        return jnp.sum(self.up_weights*jnp.sum(self.down_weights*per_leg_state,axis=(-1,-2)),-1)

    def entropy(self, seed):
        # Exact sum over gate choices; pathwise MC for Gaussian tanh Jacobians.
        sample=self.loc+self.scale*jax.random.normal(seed,self.loc.shape)
        jac=2*(jnp.log(2.)-sample-jax.nn.softplus(-2*sample))
        motor=jnp.sum(.5*jnp.log(2*jnp.pi*jnp.e)+jnp.log(self.scale)+jac,-1)
        value=self.expectation(motor)
        if self.routing=='asym':
            hu=-bernoulli_log_prob(self.up,self.up_prob)
            hd=-bernoulli_log_prob(self.down,self.down_prob)
            value+=jnp.sum(hu,-1)+jnp.sum(self.up_weights*jnp.sum(hd,-1),-1)
        return value

    def kl_divergence(self, old):
        motor=jnp.sum(jnp.log(self.scale/old.scale)+(old.scale**2+(old.loc-self.loc)**2)/(2*self.scale**2)-.5,-1)
        value=old.expectation(motor)
        if self.routing=='asym':
            ku=bernoulli_log_prob(old.up,old.up_prob)-bernoulli_log_prob(self.up,old.up_prob)
            kd=bernoulli_log_prob(old.down,old.down_prob)-bernoulli_log_prob(self.down,old.down_prob)
            value+=jnp.sum(ku,-1)+jnp.sum(old.up_weights*jnp.sum(kd,-1),-1)
        else:
            value=jnp.where(jnp.all(old.fixed==self.fixed,-1),value,jnp.inf)
        return value

    def expected_rates(self):
        return jnp.mean(self.up_prob,-1),jnp.sum(self.up_weights*jnp.mean(self.down_prob,-1),-1)


class HybridDistribution(distribution.ParametricDistribution):
    def __init__(self,routing,actuator_indices):
        super().__init__(844,distribution.IdentityPostprocessor(),1,False)
        self.routing=routing; self.actuator_indices=actuator_indices

    def create_dist(self,parameters):
        return CausalDistribution(parameters,self.routing,self.actuator_indices)

    def postprocess(self,action):
        return jnp.concatenate((action[...,:8],jnp.tanh(action[...,8:])),axis=-1)

    def inverse_postprocess(self,action):
        return jnp.concatenate((action[...,:8],jnp.arctanh(action[...,8:])),axis=-1)

    def log_prob(self,parameters,actions):
        motor=actions[...,8:]
        jac=jnp.sum(2*(jnp.log(2.)-motor-jax.nn.softplus(-2*motor)),-1)
        return self.create_dist(parameters).log_prob(actions)-jac

    def entropy(self,parameters,seed):
        return self.create_dist(parameters).entropy(seed)


def make_sparse_components(observation_size,action_size,*,mapping,architecture,budget,routing,rates=None,
                           preprocess_observations_fn=types.identity_observation_preprocessor,**factory):
    if action_size!=20: raise ValueError('Expected hybrid action size 20')
    central=make_central_networks(observation_size,12,preprocess_observations_fn=preprocess_observations_fn,**factory)
    module=SparseActor(*(tuple(tuple(int(i) for i in row) for row in mapping[k])
                        for k in ('actuator_indices','qpos_relative_indices','qvel_relative_indices')),
                       budget=budget,local_widths=tuple(architecture['local_widths']),
                       global_widths=tuple(architecture['global_widths']),motor_widths=tuple(architecture['motor_widths']))
    rates=tuple(rates or (budget,budget))
    def actor_obs(normalizer,observation):
        return preprocess_observations_fn(observation['state'],networks.normalizer_select(normalizer,'state'))
    def apply(normalizer,params,observation):
        table=module.apply(params,actor_obs(normalizer,observation))
        clock=observation['route_clock'][...,0]
        flags=periodic_flags(clock,rates)
        return jnp.concatenate((table,flags),-1)
    def diagnostic(normalizer,params,observation,up,down,**kwargs):
        return module.apply(params,actor_obs(normalizer,observation),up_gates=up,down_gates=down,**kwargs)
    policy=networks.FeedForwardNetwork(init=lambda rng:module.init(rng,jnp.zeros((1,48))),apply=apply)
    value=networks.FeedForwardNetwork(init=central.value_network.init,
        apply=lambda n,p,o:central.value_network.apply(n,p,{k:o[k] for k in ('state','privileged_state')}))
    return PPONetworks(policy,value,HybridDistribution(routing,module.actuator_indices)),diagnostic


def make_sparse_networks(observation_size,action_size,**kwargs):
    return make_sparse_components(observation_size,action_size,**kwargs)[0]


def budgeted_loss(original,coefficient,budget,routing):
    def loss(params,normalizer_params,data,rng,ppo_network,**kwargs):
        total,metrics=original(params,normalizer_params,data,rng,ppo_network,**kwargs)
        logits=ppo_network.policy_network.apply(normalizer_params,params.policy,data.observation)
        up,down=ppo_network.parametric_action_distribution.create_dist(logits).expected_rates()
        penalty=coefficient*jnp.mean((up-budget)**2+(down-budget)**2) if routing=='asym' else jnp.asarray(0.)
        return total+penalty,{**metrics,'total_loss':total+penalty,'communication_budget_penalty':penalty,
                              'expected_up_fraction':up.mean(),'expected_down_fraction':down.mean()}
    return loss
