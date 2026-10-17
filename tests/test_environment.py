"""Physical Go1/ownership acceptance tests; execute only on Slurm compute nodes.

Tolerances fixed before execution: float32 force rtol=2e-5, atol=2e-5 N m;
state/observation parity rtol=2e-5, atol=2e-6. Indexing, frozen sensors, immutable
predecessors and field ownership use exact equality. No training is performed.
"""

import copy

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest
from mujoco_playground._src.locomotion.go1.joystick import Joystick, default_config

from neuroroute.environment import (
    FaultGo1, JOINT_ROLES, LEGS, LegMap, assemble_actions, scale_actuators,
    split_observations,
)

STATE_ATOL = 2e-6
STATE_RTOL = 2e-5
FORCE_ATOL = 2e-5
FORCE_RTOL = 2e-5


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=STATE_ATOL, rtol=STATE_RTOL)


@pytest.fixture(scope="module")
def environments():
    config = default_config()
    config.impl = "jax"
    config.pert_config.enable = False
    config.noise_config.level = 0.0
    official = Joystick(task="flat_terrain", config=config)
    intact = FaultGo1(noise_level=0.0)
    sensor = FaultGo1(fault_kind="sensor", leg=2, onset_s=0.04,
                     freeze_s=0.06, noise_level=0.0)
    return official, intact, sensor


@pytest.fixture(scope="module")
def reset_states(environments):
    # Legacy PRNGKey also permits exact host snapshots of all info leaves.
    key = jax.random.PRNGKey(847)
    return tuple(jax.jit(env.reset)(key) for env in environments)


@pytest.fixture(scope="module")
def steps(environments):
    return tuple(jax.jit(env.step) for env in environments)


def test_discovered_joint_mapping_matches_mujoco_addresses(environments):
    _, env, _ = environments
    model, mapping = env.mj_model, env.leg_map
    np.testing.assert_array_equal(np.sort(mapping.actuator.ravel()), np.arange(12))
    np.testing.assert_array_equal(np.sort(mapping.qpos_local.ravel()), np.arange(12))
    np.testing.assert_array_equal(np.sort(mapping.qvel_local.ravel()), np.arange(12))
    for i, leg in enumerate(LEGS):
        for k, role in enumerate(JOINT_ROLES):
            actuator_id = mapping.actuator[i, k]
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            name = model.joint(joint_id).name
            assert name.startswith(leg + "_") and role in name
            assert model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
            assert model.jnt_qposadr[joint_id] == 7 + mapping.qpos_local[i, k]
            assert model.jnt_dofadr[joint_id] == 6 + mapping.qvel_local[i, k]
    values = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    assembled = assemble_actions(values, mapping)
    assert assembled.shape == (2, 12)
    for i in range(4):
        np.testing.assert_array_equal(assembled[:, mapping.actuator[i]], values[:, i])


def test_mapping_and_action_assembly_do_not_assume_actuator_order():
    # A deliberately different actuator/qpos/qvel order catches hard-coded
    # slices even if the installed XML happens to have an accommodating order.
    actuator = np.array([[5, 1, 9], [3, 7, 0], [11, 4, 8], [2, 10, 6]])
    qpos = np.roll(actuator, 1, axis=0)
    qvel = np.roll(actuator, 2, axis=0)
    mapping = LegMap(actuator, qpos, qvel, None)
    actor = jnp.arange(96, dtype=jnp.float32).reshape(2, 48)
    global_obs, local_obs = split_observations({"state": actor}, mapping)
    np.testing.assert_array_equal(global_obs, np.asarray(actor)[:, list(range(9)) + [45, 46, 47]])
    for i in range(4):
        ids = np.concatenate((9 + qpos[i], 21 + qvel[i], 33 + actuator[i]))
        np.testing.assert_array_equal(local_obs[:, i], np.asarray(actor)[:, ids])
    motor = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    assembled = np.asarray(assemble_actions(motor, mapping))
    expected = np.empty((2, 12), dtype=np.float32)
    for b in range(2):
        for i in range(4):
            for k in range(3):
                expected[b, actuator[i, k]] = np.asarray(motor)[b, i, k]
    np.testing.assert_array_equal(assembled, expected)


def test_actor_ownership_excludes_privileged_command_and_other_legs(environments):
    mapping = environments[1].leg_map
    actor = jnp.arange(48, dtype=jnp.float32)
    base = {"state": actor, "privileged_state": jnp.zeros(128)}
    global_obs, local_obs = split_observations(base, mapping)
    poisoned = split_observations({**base, "privileged_state": jnp.full(128, jnp.nan)}, mapping)
    for observed, expected in zip(poisoned, (global_obs, local_obs)):
        np.testing.assert_array_equal(observed, expected)
    changed = split_observations({"state": actor.at[45:48].add(1000)}, mapping)
    np.testing.assert_array_equal(changed[1], local_obs)
    assert not np.array_equal(changed[0], global_obs)
    for leg in range(4):
        private_ids = np.concatenate((9 + mapping.qpos_local[leg],
                                      21 + mapping.qvel_local[leg],
                                      33 + mapping.actuator[leg]))
        glob, local = split_observations({"state": actor.at[private_ids].add(1000)}, mapping)
        np.testing.assert_array_equal(glob, global_obs)
        others = np.array([i for i in range(4) if i != leg])
        np.testing.assert_array_equal(local[others], local_obs[others])
        assert not np.array_equal(local[leg], local_obs[leg])


def test_nofault_adapter_matches_official_reset_and_step(reset_states, steps):
    official, intact, _ = reset_states
    for field in ("qpos", "qvel", "actuator_force", "sensordata"):
        assert_close(getattr(intact.data, field), getattr(official.data, field))
    for key in official.obs:
        assert_close(intact.obs[key], official.obs[key])
    for t in range(3):
        action = jnp.sin(jnp.arange(12) * 0.3 + t) * 0.15
        official = steps[0](official, action)
        intact = steps[1](intact, action)
        for field in ("qpos", "qvel", "actuator_force", "sensordata"):
            assert_close(getattr(intact.data, field), getattr(official.data, field))
        for key in official.obs:
            assert_close(intact.obs[key], official.obs[key])
        assert_close(intact.reward, official.reward)
        np.testing.assert_array_equal(intact.done, official.done)
        for key in official.info:
            assert_close(intact.info[key], official.info[key])
        for key in official.metrics:
            assert_close(intact.metrics[key], official.metrics[key])
        assert int(intact.info["nr_step"]) == t + 1
        assert not bool(intact.info["nr_fault_active"])


def test_sensor_freeze_changes_only_owned_actor_qdq(environments, reset_states, steps):
    intact, frozen = reset_states[1:]
    sensor = environments[2]
    indices = np.asarray(sensor.sensor_indices)
    other_indices = np.setdiff1d(np.arange(48), indices)
    capture = None
    difference_seen = False
    for t in range(1, 7):
        action = jnp.sin(jnp.arange(12) * 0.4 + t) * 0.25
        intact, frozen = steps[1](intact, action), steps[2](frozen, action)
        for field in ("qpos", "qvel", "actuator_force", "sensordata"):
            assert_close(getattr(frozen.data, field), getattr(intact.data, field))
        assert_close(frozen.reward, intact.reward)
        np.testing.assert_array_equal(frozen.done, intact.done)
        assert_close(frozen.obs["privileged_state"], intact.obs["privileged_state"])
        np.testing.assert_array_equal(frozen.obs["state"][other_indices],
                                      intact.obs["state"][other_indices])
        if t == sensor.onset_step:
            capture = np.asarray(intact.obs["state"])[indices].copy()
        active = sensor.onset_step <= t < sensor.onset_step + sensor.freeze_steps
        assert bool(frozen.info["nr_fault_active"]) == active
        if active:
            np.testing.assert_array_equal(frozen.obs["state"][indices], capture)
            difference_seen |= not np.array_equal(capture, np.asarray(intact.obs["state"])[indices])
        else:
            np.testing.assert_array_equal(frozen.obs["state"], intact.obs["state"])
    assert difference_seen, "Probe trajectory must actually change measured joints during freeze"


@pytest.fixture(scope="module")
def force_probes(environments, reset_states):
    env = environments[1]
    model = env.mjx_model
    # Forward evaluates force at identical q,dq,ctrl; stepping would confound
    # changed force with changed dynamics. Compile once for all scales/legs.
    forward = jax.jit(mjx.forward)
    data = reset_states[1].data
    qvel = data.qvel.at[6:].set(jnp.linspace(-0.2, 0.2, 12))
    data = data.replace(qvel=qvel)
    neutral = np.asarray(data.qpos)[7:]
    controls = np.asarray(env.mj_model.actuator_ctrlrange)
    # Small position errors exercise unsaturated PD force.
    unsaturated = neutral + np.linspace(0.004, 0.012, 12)
    # Pick the farther legal target per actuator to ensure active saturation.
    saturated = np.where(abs(controls[:, 0] - neutral) > abs(controls[:, 1] - neutral),
                         controls[:, 0], controls[:, 1])
    probes = []
    for ctrl in (unsaturated, saturated):
        fixed = data.replace(ctrl=jnp.asarray(ctrl))
        nominal = forward(model, fixed)
        probes.append((fixed, nominal))
    return env, forward, probes


@pytest.mark.parametrize("leg", range(4))
@pytest.mark.parametrize("retained", [0.0, 0.5, 1.0])
def test_actual_mjx_actuator_force_scales_at_fixed_state(force_probes, leg, retained):
    env, forward, probes = force_probes
    mapping, model = env.leg_map, env.mjx_model
    selected = mapping.actuator[leg]
    other = np.setdiff1d(np.arange(12), selected)
    gain_before = np.asarray(model.actuator_gainprm).copy()
    damping_before = np.asarray(model.dof_damping).copy()
    scales = jnp.ones(12).at[selected].set(retained)
    scaled_model = scale_actuators(model, scales)
    for i, (fixed, nominal) in enumerate(probes):
        force = np.asarray(nominal.actuator_force)
        ranges = np.asarray(model.actuator_forcerange)
        boundary = np.isclose(force, ranges[:, 0], atol=FORCE_ATOL) | np.isclose(force, ranges[:, 1], atol=FORCE_ATOL)
        if i == 0:
            assert not np.any(boundary), "Unsaturated probe accidentally reached a force limit"
            assert np.all(abs(force[selected]) > 1e-4), "Nonzero force needed to test scaling"
        else:
            assert np.all(boundary[selected]), "Saturated probe must hit all selected actuator limits"
        faulted = forward(scaled_model, fixed)
        np.testing.assert_allclose(np.asarray(faulted.actuator_force)[selected],
                                   retained * force[selected], atol=FORCE_ATOL, rtol=FORCE_RTOL)
        np.testing.assert_allclose(np.asarray(faulted.actuator_force)[other], force[other],
                                   atol=FORCE_ATOL, rtol=FORCE_RTOL)
        np.testing.assert_array_equal(faulted.qpos, fixed.qpos)
        np.testing.assert_array_equal(faulted.qvel, fixed.qvel)
        np.testing.assert_array_equal(faulted.ctrl, fixed.ctrl)
    np.testing.assert_array_equal(model.actuator_gainprm, gain_before)
    # Kd is passive dof_damping in this Go1 model: actuator attenuation must
    # not silently attenuate passive mechanics or unaffected motors.
    np.testing.assert_array_equal(scaled_model.dof_damping, damping_before)


def test_step_does_not_mutate_predecessor_python_containers(environments, reset_states, monkeypatch):
    env, state = environments[1], reset_states[1]
    info_before = jax.tree.map(lambda x: np.asarray(x).copy(), state.info)
    metrics_before = jax.tree.map(lambda x: np.asarray(x).copy(), state.metrics)

    def mutating_upstream(view, incoming, action):
        # Exercise the actual Python alias boundary without another eager
        # physics compilation; upstream has these exact mutation categories.
        incoming.info["last_act"] = action
        incoming.info["feet_air_time"] += 17.0
        incoming.metrics["swing_peak"] = jnp.array(19.0)
        return incoming.replace(reward=jnp.array(1.0))

    monkeypatch.setattr(Joystick, "step", mutating_upstream)
    result = env.step(state, jnp.ones(12))
    assert result.info is not state.info
    assert result.metrics is not state.metrics
    for key, value in info_before.items():
        np.testing.assert_array_equal(state.info[key], value)
    for key, value in metrics_before.items():
        np.testing.assert_array_equal(state.metrics[key], value)
    assert int(result.info["nr_step"]) == int(state.info["nr_step"]) + 1


def test_actuator_fault_model_is_used_at_declared_step(environments, reset_states, monkeypatch):
    env = copy.copy(environments[1])
    env.fault_kind, env.fault_leg, env.retained, env.onset_step = "actuator", 1, 0.5, 2
    models_seen = []

    def capture_model(view, incoming, action):
        models_seen.append(view.mjx_model)
        return incoming

    monkeypatch.setattr(Joystick, "step", capture_model)
    state = reset_states[1]
    for nr_step in (1, 2):
        incoming = state.replace(info={**state.info, "nr_step": jnp.array(nr_step)})
        outgoing = env.step(incoming, jnp.zeros(12))
        expected = np.ones(12)
        if nr_step >= env.onset_step:
            expected[env.leg_map.actuator[1]] = 0.5
        actual_gain = np.asarray(models_seen[-1].actuator_gainprm[:, 0])
        np.testing.assert_allclose(actual_gain, expected * np.asarray(env.mjx_model.actuator_gainprm[:, 0]),
                                   atol=FORCE_ATOL, rtol=FORCE_RTOL)
        assert bool(outgoing.info["nr_fault_active"]) == (nr_step >= 2)
    np.testing.assert_array_equal(env.mjx_model.actuator_gainprm,
                                  environments[1].mjx_model.actuator_gainprm)


def test_forked_actual_step_leaves_predecessor_unchanged(reset_states, steps):
    state = reset_states[1]
    before_q = np.asarray(state.data.qpos).copy()
    before_v = np.asarray(state.data.qvel).copy()
    before_obs = np.asarray(state.obs["state"]).copy()
    first = steps[1](state, jnp.full(12, 0.1))
    second = steps[1](state, jnp.full(12, -0.1))
    np.testing.assert_array_equal(state.data.qpos, before_q)
    np.testing.assert_array_equal(state.data.qvel, before_v)
    np.testing.assert_array_equal(state.obs["state"], before_obs)
    assert int(state.info["nr_step"]) == 0
    assert not np.array_equal(first.data.qvel, second.data.qvel)
