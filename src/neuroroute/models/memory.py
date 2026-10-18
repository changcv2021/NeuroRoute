"""Receiver message state and packet ages."""
import jax.numpy as jnp

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
