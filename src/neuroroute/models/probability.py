"""Stable KL diagnostics; preserve the actor sampling and training likelihood."""
import jax
import jax.numpy as jnp
from .routing import CausalDistribution, up_masks, select_branch
from .flycns import make_components as base_components, CurriculumDistribution, CurriculumHybrid

def stable_gate(logit,floor,full):
    floor=jnp.where(full,0.,floor)
    log_floor=jnp.where(floor>0,jnp.log(jnp.maximum(floor,jnp.finfo(jnp.float32).tiny)),-jnp.inf)
    log_rest=jnp.log1p(-floor)
    log_q=log_rest+jax.nn.log_sigmoid(-logit)
    # Negative logits: mixture logsumexp remains finite even for tiny p.
    mix=jnp.logaddexp(log_floor,log_rest+jax.nn.log_sigmoid(logit))
    # Positive logits: avoid cancellation in logsumexp near p=1.
    # Clamp only the unused branch's argument to avoid0*infinite AD terms.
    complement=jnp.log1p(-jnp.exp(jnp.minimum(log_q,-jnp.log(2.))))
    log_p=jnp.where(logit>0,complement,mix)
    log_p=jnp.where(full,0.,log_p);log_q=jnp.where(full,0.,log_q)
    p=jnp.where(full,1.,jnp.exp(log_p));q=jnp.where(full,0.,jnp.exp(log_q))
    return log_p,log_q,p,q

class StableCurriculumDistribution(CausalDistribution):
    def __init__(self,parameters,actuator_indices,budget):
        super().__init__(parameters[...,:844],'asym',actuator_indices)
        self.parameters=parameters
        floor=jnp.clip((parameters[...,844]-budget)/(1-budget),0.,1.)
        self.full=floor>=1.
        self.up_logp,self.up_logq,self.up_prob,uq=stable_gate(self.up,floor[...,None],self.full[...,None])
        self.down_logp,self.down_logq,self.down_prob,dq=stable_gate(self.down,floor[...,None,None],self.full[...,None,None])
        self.up_complement=uq;self.down_complement=dq
        self.up_weights=jnp.prod(jnp.where(up_masks()==1,self.up_prob[...,None,:],uq[...,None,:]),-1)
        self.down_weights=jnp.stack((dq,self.down_prob),-1)

    def log_prob(self,action):
        up,down,motor=action[...,:4],action[...,4:8],action[...,8:]
        loc,scale=self.selected_motor(up,down)
        normal=jnp.sum(-.5*((motor-loc)/scale)**2-jnp.log(scale)-.5*jnp.log(2*jnp.pi),-1)
        gate=jnp.sum(up*self.up_logp+(1-up)*self.up_logq+
            down*select_branch(self.down_logp,up)+(1-down)*select_branch(self.down_logq,up),-1)
        invalid=self.full & jnp.any(action[...,:8]!=1.,-1)
        return normal+jnp.where(invalid,-jnp.inf,gate)

    def entropy(self,seed):
        sample=self.loc+self.scale*jax.random.normal(seed,self.loc.shape)
        jac=2*(jnp.log(2.)-sample-jax.nn.softplus(-2*sample))
        motor=jnp.sum(.5*jnp.log(2*jnp.pi*jnp.e)+jnp.log(self.scale)+jac,-1)
        hu=-(self.up_prob*self.up_logp+self.up_complement*self.up_logq)
        hd=-(self.down_prob*self.down_logp+self.down_complement*self.down_logq)
        return self.expectation(motor)+jnp.sum(hu,-1)+jnp.sum(self.up_weights*jnp.sum(hd,-1),-1)

    def kl_divergence(self,old):
        motor=jnp.sum(jnp.log(self.scale/old.scale)+(old.scale**2+(old.loc-self.loc)**2)/(2*self.scale**2)-.5,-1)
        ku=old.up_prob*(old.up_logp-self.up_logp)+old.up_complement*(old.up_logq-self.up_logq)
        kd=old.down_prob*(old.down_logp-self.down_logp)+old.down_complement*(old.down_logq-self.down_logq)
        value=old.expectation(motor)+jnp.sum(ku,-1)+jnp.sum(old.up_weights*jnp.sum(kd,-1),-1)
        return jnp.where(self.full & ~old.full,jnp.inf,value)

class DiagnosticKLDistribution(CurriculumDistribution):
    def __init__(self,parameters,actuator_indices,budget):
        super().__init__(parameters,actuator_indices,budget)
        self.budget=budget

    def kl_divergence(self,old):
        current=StableCurriculumDistribution(self.parameters,self.aa,self.budget)
        previous=StableCurriculumDistribution(old.parameters,old.aa,old.budget)
        return current.kl_divergence(previous)

class DiagnosticKLHybrid(CurriculumHybrid):
    def create_dist(self,parameters):
        return DiagnosticKLDistribution(parameters,self.actuator_indices,self.budget)

def make_components(observation_size,action_size,**kwargs):
    network,diagnostic=base_components(observation_size,action_size,**kwargs)
    if kwargs['variant']['cache']:
        old=network.parametric_action_distribution
        network=network.replace(parametric_action_distribution=DiagnosticKLHybrid(old.actuator_indices,old.budget))
    return network,diagnostic

def make_networks(observation_size,action_size,**kwargs):
    return make_components(observation_size,action_size,**kwargs)[0]
