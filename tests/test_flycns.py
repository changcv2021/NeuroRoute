"""Behavioral contracts for final-model communication and budget scheduling."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuroroute.models.common import transmit
from neuroroute.models.flycns import CachedActor
from neuroroute.models.memory import initial_cache
from neuroroute.models.probability import DiagnosticKLDistribution
from neuroroute.training.curriculum import curriculum_target


def test_only_sent_packets_cost_bits():
    delivered, wire = transmit(jnp.ones((4, 32)), jnp.array([True, False, True, False]))
    assert int(wire['total_bits'].sum()) == 2 * (32 * 32 + 4)
    np.testing.assert_array_equal(delivered[jnp.array([1, 3]), :], 0)


@pytest.mark.parametrize('budget,targets', [
    (.75, [1., .9, .8, .75]), (.5, [1., .8, .65, .5]), (.25, [1., .75, .5, .25]),
])
def test_curriculum_boundaries_and_disabled_mode(budget, targets):
    starts = [0, 45875200, 91750400, 137625600]
    for stage, step in enumerate(starts):
        assert float(curriculum_target(step, budget)) == pytest.approx(targets[stage])
        if stage:
            assert float(curriculum_target(step - 1, budget)) == pytest.approx(targets[stage - 1])
        assert float(curriculum_target(step, budget, False)) == pytest.approx(budget)


@pytest.fixture(scope='module')
def actor_setup():
    indices = tuple(tuple(range(3 * i, 3 * i + 3)) for i in range(4))
    model = CachedActor(indices, indices, indices, budget=.25)
    actor = jnp.linspace(-.2, .2, 48)[None, :]
    cache = initial_cache((1,)).at[:, :256].set(jnp.linspace(-1., 1., 256)[None, :])
    cache = cache.at[:, 256:].set(7.)
    params = model.init(jax.random.PRNGKey(17), actor, cache)
    return model, params, actor, cache


def test_closed_gates_retain_payloads_and_increment_age(actor_setup):
    model, params, actor, cache = actor_setup
    gates = jnp.zeros((1, 4))
    _, route = model.apply(params, actor, cache, up_gates=gates, down_gates=gates)
    np.testing.assert_array_equal(route['next_communication_state'][:, :256], cache[:, :256])
    np.testing.assert_array_equal(route['next_communication_state'][:, 256:], 8.)
    assert int(route['up_wire']['total_bits'].sum() + route['down_wire']['total_bits'].sum()) == 0


def test_no_age_variant_is_invariant_to_age_values(actor_setup):
    model, params, actor, cache = actor_setup
    no_age = model.clone(use_age=False)
    changed_age = cache.at[:, 256:].set(700.)
    np.testing.assert_array_equal(no_age.apply(params, actor, cache), no_age.apply(params, actor, changed_age))


def test_stable_kl_for_saturated_gate_logits():
    indices = tuple(tuple(range(3 * i, 3 * i + 3)) for i in range(4))
    logits = jnp.zeros((1, 845)).at[:, :68].set(1000.).at[:, 844].set(.25)
    old = DiagnosticKLDistribution(logits, indices, .25)
    divergence = DiagnosticKLDistribution(logits, indices, .25).kl_divergence(old)
    np.testing.assert_allclose(divergence, 0., atol=1e-7)
    gradient = jax.grad(lambda x: DiagnosticKLDistribution(x, indices, .25).kl_divergence(old).sum())(logits)
    assert bool(jnp.isfinite(gradient).all())
