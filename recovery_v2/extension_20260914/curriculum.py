"""Only the new 25% schedule extends the frozen science implementation."""
import jax.numpy as jnp
from neuroroute.flycns_v2 import curriculum_target as original_target


def curriculum_target(steps, budget, enabled=True):
    if budget in (.75, .5):
        return original_target(steps, budget, enabled)
    if budget != .25:
        raise ValueError('Unregistered budget')
    if not enabled:
        return jnp.asarray(budget, jnp.float32)
    stage = jnp.sum(jnp.asarray(steps) >= jnp.asarray((45875200, 91750400, 137625600)))
    return jnp.asarray((1., .75, .5, .25), jnp.float32)[stage]


def install():
    # This is the sole trainer reference to curriculum_target. No file is edited.
    from recovery_v2 import ppo_train
    if ppo_train.curriculum_target not in (original_target, curriculum_target):
        raise ValueError('Unexpected trainer curriculum implementation')
    ppo_train.curriculum_target = curriculum_target
    return ppo_train
