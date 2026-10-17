"""Protocol acceptance T04–T08/T22; execute under a compute allocation."""

import jax.numpy as jnp
import numpy as np
import pytest

from neuroroute.communication import (
    advance, bit_cost, empty_state, periodic_gates, reset_state,
)


@pytest.mark.parametrize("count, expected", [(0, 8), (1, 72), (8, 520)])
def test_t04_t05_t06_exact_wire_cost(count, expected):
    gates = jnp.arange(8) < count
    cost = bit_cost(gates[:4], gates[4:])
    assert int(cost.total) == expected
    assert int(cost.up + cost.down) == expected


def test_t07_closed_gate_retains_bytes_and_ages_in_seconds():
    state = empty_state()
    sent = jnp.ones(4, dtype=bool)
    closed = ~sent
    first = jnp.arange(32, dtype=jnp.uint8).reshape(4, 8)
    state = advance(state, sent, first, sent, first + 40, 0.02)
    for _ in range(3):
        state = advance(state, closed, first + 80, closed, first + 100, 0.02)
    np.testing.assert_array_equal(state.up.payload, first)
    np.testing.assert_array_equal(state.down.payload, first + 40)
    np.testing.assert_allclose(state.up.age, 0.06, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(state.up.valid, True)


def test_t08_interleaved_episode_reset_is_batch_local():
    state = empty_state((3,))
    flags = jnp.ones((3, 4), dtype=bool)
    packets = jnp.arange(96, dtype=jnp.uint8).reshape(3, 4, 8) + 1
    state = advance(state, flags, packets, flags, packets, 0.02)
    state = advance(state, ~flags, packets + 1, ~flags, packets + 1, 0.02)
    reset = reset_state(state, jnp.array([False, True, False]))
    for direction in (reset.up, reset.down):
        np.testing.assert_array_equal(direction.payload[1], 0)
        np.testing.assert_array_equal(direction.valid[1], False)
        np.testing.assert_array_equal(direction.age[1], 0.0)
        selected = jnp.array([0, 2])
        np.testing.assert_array_equal(direction.payload[selected], packets[selected])
        np.testing.assert_array_equal(direction.valid[selected], True)
        np.testing.assert_allclose(direction.age[selected], 0.02)


@pytest.mark.parametrize("budget", [0.25, 0.5, 1.0])
def test_t22_round_robin_cost_recomputed_from_packet_records(budget):
    gates = periodic_gates(jnp.arange(512), budget)
    # Independent accounting from a hypothetical packet log, with no bit_cost.
    packet_log = np.asarray(gates, dtype=bool)
    independently_counted = sum(1 + (8 * 8 if sent else 0)
                                for row in packet_log for sent in row)
    assert independently_counted / len(packet_log) == 520 * budget
    measured = bit_cost(gates[:, :4], gates[:, 4:])
    assert int(jnp.sum(measured.total)) == independently_counted
    # No direction or leg preference is introduced by the periodic baseline.
    np.testing.assert_array_equal(packet_log.sum(axis=0), packet_log.sum() // 8)


def test_periodic_schedule_depends_only_on_public_step_and_wraps():
    np.testing.assert_array_equal(periodic_gates(jnp.arange(512), 0.25),
                                  periodic_gates(jnp.arange(512) + 512, 0.25))
