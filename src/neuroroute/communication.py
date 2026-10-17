"""Eight-slot logical link: four uplinks followed by four downlinks.

Each slot always costs one flag bit; a sent packet adds eight uint8 symbols.
All arrays accept arbitrary leading environment/trajectory batch dimensions.
"""

from typing import NamedTuple

import jax.numpy as jnp
from flax import struct

N_LEGS = 4
PACKET_BYTES = 8
FULL_BITS = 520


@struct.dataclass
class PacketCache:
    payload: jnp.ndarray
    age: jnp.ndarray
    valid: jnp.ndarray


@struct.dataclass
class CommunicationState:
    up: PacketCache
    down: PacketCache


class BitCost(NamedTuple):
    total: jnp.ndarray
    up: jnp.ndarray
    down: jnp.ndarray


def empty_state(batch_shape=()) -> CommunicationState:
    """Initial/reset cache. Ages are seconds, not steps."""
    shape = tuple(batch_shape) + (N_LEGS,)
    cache = PacketCache(
        jnp.zeros(shape + (PACKET_BYTES,), dtype=jnp.uint8),
        jnp.zeros(shape, dtype=jnp.float32),
        jnp.zeros(shape, dtype=jnp.bool_),
    )
    return CommunicationState(up=cache, down=cache)


def reset_state(state: CommunicationState, reset_mask) -> CommunicationState:
    """Clear only environments whose episode ended (including truncations)."""
    mask = jnp.asarray(reset_mask, dtype=jnp.bool_)

    def clear(cache):
        return PacketCache(
            jnp.where(mask[..., None, None], 0, cache.payload),
            jnp.where(mask[..., None], 0.0, cache.age),
            jnp.where(mask[..., None], False, cache.valid),
        )

    return CommunicationState(clear(state.up), clear(state.down))


def tick(state: CommunicationState, dt) -> CommunicationState:
    """Advance receiver-local clocks exactly once per control step."""
    elapsed = jnp.asarray(dt, dtype=jnp.float32)[..., None]
    return CommunicationState(
        state.up.replace(age=state.up.age + elapsed),
        state.down.replace(age=state.down.age + elapsed),
    )


def deliver(cache: PacketCache, gate, payload) -> PacketCache:
    """A closed gate leaves bytes unchanged, even for arbitrary placeholders."""
    gate = jnp.asarray(gate, dtype=jnp.bool_)
    return PacketCache(
        jnp.where(gate[..., None], jnp.asarray(payload, jnp.uint8), cache.payload),
        jnp.where(gate, 0.0, cache.age),
        cache.valid | gate,
    )


def advance(state, up_gate, up_payload, down_gate, down_payload, dt):
    state = tick(state, dt)
    return CommunicationState(
        deliver(state.up, up_gate, up_payload),
        deliver(state.down, down_gate, down_payload),
    )


def bit_cost(up_gate, down_gate) -> BitCost:
    """Audit cost from the actual transmitted flags, never from gate scores."""
    up = 4 + 64 * jnp.sum(jnp.asarray(up_gate, jnp.int32), axis=-1)
    down = 4 + 64 * jnp.sum(jnp.asarray(down_gate, jnp.int32), axis=-1)
    return BitCost(up + down, up, down)


def cache_features(cache: PacketCache):
    """Receiver-visible bytes, validity, and local age; no sender features."""
    decoded = cache.payload.astype(jnp.float32) / 127.5 - 1.0
    decoded = jnp.where(cache.valid[..., None], decoded, 0.0)
    return jnp.concatenate(
        (decoded, cache.valid[..., None].astype(jnp.float32),
         jnp.log1p(cache.age)[..., None]), axis=-1,
    )


def full_gates(batch_shape=()):
    return jnp.ones(tuple(batch_shape) + (8,), dtype=jnp.bool_)


def periodic_gates(step, budget_fraction=0.25):
    """Public round-robin schedule with the flag-inclusive mean budget.

    Over each 64-step cycle the schedule sends exactly (520*b - 8)
    packets, equally rotating through the eight directed links. No private
    score enters this schedule. Fractions are protocol settings, hence static.
    """
    if budget_fraction not in (0.25, 0.5, 1.0):
        raise ValueError("Only locked budget fractions 0.25, 0.5, 1.0 are supported")
    numerator = int(FULL_BITS * budget_fraction - 8)
    # Reduce step modulo the 512-step joint cost/link cycle to avoid overflow.
    t = jnp.asarray(step, jnp.int32) % 512
    before = (t * numerator) // 64
    count = ((t + 1) * numerator) // 64 - before
    slot = jnp.arange(8, dtype=jnp.int32)
    return ((slot - before[..., None]) % 8) < count[..., None]
