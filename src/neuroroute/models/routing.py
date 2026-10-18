"""Causal gate distributions and conditional motor probabilities."""
import jax
import jax.numpy as jnp
from brax.training import distribution

def up_masks():
    return ((jnp.arange(16)[:, None] >> jnp.arange(4)) & 1).astype(jnp.float32)

def select_branch(values, gates):
    """Select the 16-way axis immediately after the batch axes."""
    index = jnp.sum(gates.astype(jnp.int32) * (1 << jnp.arange(4)), axis=-1)
    extra = values.ndim - gates.ndim
    selection = index.reshape(index.shape + (1,) * (extra + 1))
    return jnp.take_along_axis(values, selection, axis=index.ndim).squeeze(index.ndim)

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
