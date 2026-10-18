"""Communication budget curricula for the final FlyCNS implementation."""
import jax.numpy as jnp


def curriculum_target(steps, budget, enabled=True):
    schedules = {
        .75: (1., .9, .8, .75),
        .5: (1., .8, .65, .5),
        .25: (1., .75, .5, .25),
    }
    if budget not in schedules:
        raise ValueError('Unsupported communication budget')
    if not enabled:
        return jnp.asarray(budget, jnp.float32)
    stage = jnp.sum(jnp.asarray(steps) >= jnp.asarray((45875200, 91750400, 137625600)))
    return jnp.asarray(schedules[budget], jnp.float32)[stage]
