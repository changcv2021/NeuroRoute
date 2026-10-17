"""Isolated MATCH precision candidate: float64 actor arithmetic, float32 boundary.

Parameters, recurrent storage, PPO arrays and physics remain float32. This
candidate changes internal arithmetic precision, never the action, information,
loss, full-prefix replay definition, or acceptance tolerance. It is not released
for formal training until dtype, derivative and Go1 replay validation pass.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from .recurrent_published_policy import (
    RecurrentPublishedPolicy as FrozenPolicy, empty_recurrent_match_state,
    reset_recurrent_match_state)


def floating_dtype(tree, dtype):
    return jax.tree.map(lambda x: x.astype(dtype)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


class RecurrentPublishedPolicy(FrozenPolicy):
    @nn.compact
    def __call__(self, global_obs, local_obs, state, key, dt, *, action=None,
                 up_received=None, down_received=None, deterministic_motor=False):
        # The context is local to actor tracing; MJX and the optimizer retain
        # their existing runtime and explicit float32 data types.
        with jax.experimental.enable_x64():
            result = super().__call__(global_obs.astype(jnp.float64),
                local_obs.astype(jnp.float64), floating_dtype(state, jnp.float64),
                key, dt, action=floating_dtype(action, jnp.float64),
                up_received=up_received, down_received=down_received,
                deterministic_motor=deterministic_motor)
            return floating_dtype(result, jnp.float32)
