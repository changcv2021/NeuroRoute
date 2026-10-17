"""Information isolation and complete causal probability acceptance tests.

T09 tolerances are fixed before execution: absolute log-prob 2e-5 and ratio
2e-5 for float32. T03/T10 invariances use exact equality where appropriate.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuroroute.communication import bit_cost, empty_state
from neuroroute.policy import HybridPolicy


@pytest.fixture(scope="module")
def actor():
    model = HybridPolicy(local_width=16, global_width=24)
    global_obs = jnp.arange(12, dtype=jnp.float32).reshape(2, 6) / 12
    local_obs = jnp.arange(72, dtype=jnp.float32).reshape(2, 4, 9) / 72
    cache = empty_state((2,))
    variables = model.init(jax.random.key(0), global_obs, local_obs, cache,
                           jax.random.key(1), 0.02)
    return model, variables, global_obs, local_obs, cache


def run(actor, key=2, **kwargs):
    model, variables, global_obs, local_obs, cache = actor
    return model.apply(variables, global_obs, local_obs, cache,
                       jax.random.key(key), 0.02, **kwargs)


def test_t09_t11_replay_uses_saved_causal_actions_not_new_random_samples(actor):
    sampled = run(actor)
    replay = run(actor, key=999, action=sampled.action)
    np.testing.assert_allclose(replay.log_prob, sampled.log_prob, rtol=0, atol=2e-5)
    np.testing.assert_allclose(jnp.exp(replay.log_prob - sampled.log_prob), 1,
                               rtol=0, atol=2e-5)
    np.testing.assert_array_equal(replay.motor_action, sampled.motor_action)
    for old, new in zip(jax.tree.leaves(sampled.cache), jax.tree.leaves(replay.cache)):
        np.testing.assert_array_equal(old, new)
    # This detects omitting one of the three kinds of stochastic action.
    np.testing.assert_allclose(sampled.log_prob,
        sampled.log_prob_gate + sampled.log_prob_message + sampled.log_prob_motor,
        rtol=0, atol=2e-5)


def test_t11_saved_uplink_conditions_downlink_but_cannot_bypass_saved_downlink(actor):
    model, variables, global_obs, local_obs, cache = actor
    # Uniform uplink likelihood removes its contribution to any observed
    # likelihood difference when changing delivered upstream symbols.
    params = dict(variables["params"])
    params["up_message"] = jax.tree.map(jnp.zeros_like, params["up_message"])
    variables = {"params": params}
    full = jnp.ones((2, 8), dtype=bool)
    initial = model.apply(variables, global_obs, local_obs, cache,
                          jax.random.key(66), 0.02, forced_gates=full)
    changed_action = initial.action.replace(up_payload=255 - initial.action.up_payload)
    changed = model.apply(variables, global_obs, local_obs, cache,
                          jax.random.key(66), 0.02, forced_gates=full,
                          action=changed_action)
    assert not np.allclose(changed.log_prob_message, initial.log_prob_message,
                           rtol=0, atol=1e-5)
    np.testing.assert_array_equal(changed.motor_mean, initial.motor_mean)
    np.testing.assert_array_equal(changed.cache.down.payload, initial.cache.down.payload)


def test_t10_unsent_placeholders_have_no_probability_entropy_cost_or_cache_effect(actor):
    closed = jnp.zeros((2, 8), dtype=bool)
    sampled = run(actor, forced_gates=closed)
    changed_action = sampled.action.replace(
        up_payload=jnp.full((2, 4, 8), 255, dtype=jnp.uint8),
        down_payload=jnp.full((2, 4, 8), 173, dtype=jnp.uint8),
    )
    replay = run(actor, key=2, forced_gates=closed, action=changed_action)
    np.testing.assert_array_equal(replay.log_prob, sampled.log_prob)
    np.testing.assert_array_equal(replay.entropy, sampled.entropy)
    np.testing.assert_array_equal(replay.log_prob_message, 0.0)
    np.testing.assert_array_equal(replay.entropy_message, 0.0)
    np.testing.assert_array_equal(replay.motor_mean, sampled.motor_mean)
    np.testing.assert_array_equal(replay.cache.up.payload, 0)
    np.testing.assert_array_equal(replay.cache.down.payload, 0)
    np.testing.assert_array_equal(bit_cost(replay.action.up_gate, replay.action.down_gate).total, 8)


def test_t03_other_leg_private_input_cannot_bypass_closed_links(actor):
    model, variables, global_obs, local_obs, cache = actor
    closed = jnp.zeros((2, 8), dtype=bool)
    baseline = run(actor, forced_gates=closed)
    changed = model.apply(variables, global_obs, local_obs.at[:, 1, :].add(100), cache,
                          jax.random.key(2), 0.02, forced_gates=closed)
    np.testing.assert_array_equal(changed.motor_mean[:, [0, 2, 3]],
                                  baseline.motor_mean[:, [0, 2, 3]])
    np.testing.assert_array_equal(changed.cache.up.payload, baseline.cache.up.payload)
    # With learned gates teacher-forced closed, global downlink probabilities
    # are unchanged too. Own uplink probabilities are permitted to change.
    baseline = run(actor, action=baseline.action)
    changed = model.apply(variables, global_obs, local_obs.at[:, 1, :].add(100), cache,
                          jax.random.key(2), 0.02, action=baseline.action)
    np.testing.assert_array_equal(changed.gate_probs[:, 4:], baseline.gate_probs[:, 4:])


def test_t03_command_cannot_bypass_closed_downlink(actor):
    model, variables, global_obs, local_obs, cache = actor
    closed = jnp.zeros((2, 8), dtype=bool)
    baseline = run(actor, forced_gates=closed)
    changed = model.apply(variables, global_obs + 100, local_obs, cache,
                          jax.random.key(2), 0.02, forced_gates=closed)
    np.testing.assert_array_equal(changed.motor_mean, baseline.motor_mean)


def test_full_communication_uniform_distribution_matches_hand_density(actor):
    model, variables, global_obs, local_obs, cache = actor
    zero_params = jax.tree.map(jnp.zeros_like, variables)
    full = jnp.ones((2, 8), dtype=bool)
    output = model.apply(zero_params, global_obs, local_obs, cache,
                         jax.random.key(22), 0.02, forced_gates=full,
                         deterministic_motor=True)
    # Eight packets * eight categorical bytes; twelve unit Gaussians at zero.
    expected = -64 * math.log(256) - 6 * math.log(2 * math.pi)
    np.testing.assert_allclose(output.log_prob, expected, rtol=0, atol=5e-5)
    np.testing.assert_array_equal(output.log_prob_gate, 0.0)
    np.testing.assert_allclose(output.entropy_message, 64 * math.log(256), rtol=0, atol=5e-5)


def test_bernoulli_flags_contribute_even_when_payload_closed(actor):
    model, variables, global_obs, local_obs, cache = actor
    zeros = jax.tree.map(jnp.zeros_like, variables)
    action = run(actor, forced_gates=jnp.zeros((2, 8), dtype=bool)).action
    output = model.apply(zeros, global_obs, local_obs, cache,
                         jax.random.key(7), 0.02, action=action)
    np.testing.assert_allclose(output.log_prob_gate, -8 * math.log(2), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(output.log_prob_message, 0.0)


def test_evaluation_keeps_discrete_messages_stochastic(actor):
    full = jnp.ones((2, 8), dtype=bool)
    first = run(actor, key=50, forced_gates=full, deterministic_motor=True)
    second = run(actor, key=51, forced_gates=full, deterministic_motor=True)
    np.testing.assert_array_equal(first.action.motor_raw, first.motor_mean)
    assert not np.array_equal(first.action.up_payload, second.action.up_payload)


def test_closed_message_heads_receive_zero_likelihood_gradient(actor):
    model, variables, global_obs, local_obs, cache = actor
    closed = jnp.zeros((2, 8), dtype=bool)
    action = run(actor, forced_gates=closed).action

    def loss(params):
        output = model.apply({"params": params}, global_obs, local_obs, cache,
                             jax.random.key(0), 0.02, action=action,
                             forced_gates=closed)
        return -output.log_prob.mean()

    grads = jax.grad(loss)(variables["params"])
    for head in ("up_message", "down_message"):
        for leaf in jax.tree.leaves(grads[head]):
            np.testing.assert_array_equal(leaf, 0.0)
