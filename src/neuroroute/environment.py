"""Go1 adapter with explicit observation ownership and physical fault injection.

The official reward/step implementation is reused unchanged. A per-call model
view supplies modified actuator gains, affine biases and force limits; this is
not policy-action scaling. Observation slices are verified against the pinned
Playground source, and joint/action indices are discovered from the MJ model.
"""
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco_playground._src.locomotion.go1.joystick import Joystick, default_config

LEGS = ("FR", "FL", "RR", "RL")
JOINT_ROLES = ("hip", "thigh", "calf")


@dataclass(frozen=True)
class LegMap:
    actuator: Any
    qpos_local: Any
    qvel_local: Any
    names: Any

    @classmethod
    def from_model(cls, model):
        if (model.nq, model.nv, model.nu) != (19, 18, 12):
            raise ValueError("Go1 schema changed: expected nq=19,nv=18,nu=12")
        groups, qpos, qvel, names = [], [], [], []
        for leg in LEGS:
            aa, qq, vv, nn = [], [], [], []
            for role in JOINT_ROLES:
                matches = []
                for a in range(model.nu):
                    joint_id = int(model.actuator_trnid[a, 0])
                    name = model.joint(joint_id).name
                    if name.startswith(leg + "_") and role in name:
                        matches.append((a, joint_id, name))
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous/missing actuator for {leg}/{role}: {matches}")
                a, j, name = matches[0]
                if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                    raise ValueError("Only direct joint transmission was audited")
                aa.append(a); qq.append(int(model.jnt_qposadr[j]) - 7)
                vv.append(int(model.jnt_dofadr[j]) - 6); nn.append(name)
            groups.append(aa); qpos.append(qq); qvel.append(vv); names.append(nn)
        if sorted(sum(groups, [])) != list(range(12)):
            raise ValueError("Actuators do not form exactly four distinct legs")
        return cls(np.array(groups), np.array(qpos), np.array(qvel), names)

    def manifest(self):
        return {"legs": list(LEGS), "joint_roles": list(JOINT_ROLES),
                "actuator_indices": self.actuator.tolist(),
                "qpos_relative_indices": self.qpos_local.tolist(),
                "qvel_relative_indices": self.qvel_local.tolist(), "joint_names": self.names}


def split_observations(observation, mapping):
    """No privileged state is read here. Shapes: global[...,12], local[...,4,9]."""
    actor = observation["state"]
    if actor.shape[-1] != 48:
        raise ValueError(f"Unaudited actor observation dimension {actor.shape[-1]}")
    global_obs = jnp.concatenate((actor[..., :9], actor[..., 45:48]), axis=-1)
    local_obs = jnp.concatenate((actor[..., 9 + mapping.qpos_local],
                                 actor[..., 21 + mapping.qvel_local],
                                 actor[..., 33 + mapping.actuator]), axis=-1)
    return global_obs, local_obs


def assemble_actions(local_action, mapping):
    return jnp.zeros(local_action.shape[:-2] + (12,), dtype=local_action.dtype).at[
        ..., mapping.actuator].set(local_action)


def scale_actuators(model, scales):
    """Exactly scales fixed/affine actuator force including saturation limits."""
    return model.replace(actuator_gainprm=model.actuator_gainprm * scales[:, None],
                         actuator_biasprm=model.actuator_biasprm * scales[:, None],
                         actuator_forcerange=model.actuator_forcerange * scales[:, None])


class _ModelView:
    """A local view for one official step; never changes the shared environment."""
    def __init__(self, environment, model):
        self.environment, self.mjx_model = environment, model

    def __getattr__(self, name):
        return getattr(self.environment, name)


class FaultGo1(Joystick):
    def __init__(self, task="flat_terrain", *, impl="jax", fault_kind="none",
                 leg=0, retained=1.0, onset_s=6.0, freeze_s=0.2, noise_level=1.0):
        if fault_kind not in {"none", "actuator", "sensor"}:
            raise ValueError(fault_kind)
        if leg not in range(4) or not 0 <= retained <= 1 or onset_s < 0 or freeze_s <= 0:
            raise ValueError("Invalid fault configuration")
        config = default_config()
        config.impl = impl
        config.pert_config.enable = False
        config.noise_config.level = noise_level
        super().__init__(task=task, config=config)
        self.leg_map = LegMap.from_model(self.mj_model)
        if np.any(self.mj_model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED):
            raise ValueError("Unaudited nonlinear actuator gain")
        if np.any(self.mj_model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE):
            raise ValueError("Unaudited actuator bias type")
        if np.any(self.mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE):
            raise ValueError("Actuator activation dynamics need a separate audit")
        self.fault_kind, self.fault_leg, self.retained = fault_kind, leg, retained
        self.onset_step = round(onset_s / self.dt)
        self.freeze_steps = max(1, round(freeze_s / self.dt))
        self.sensor_indices = jnp.array(np.concatenate((9 + self.leg_map.qpos_local[leg],
                                                       21 + self.leg_map.qvel_local[leg])))

    def reset(self, rng):
        state = super().reset(rng)
        state.info["nr_step"] = jnp.array(0, dtype=jnp.int32)
        state.info["nr_frozen_sensor"] = state.obs["state"][self.sensor_indices]
        state.info["nr_fault_active"] = jnp.array(False)
        return state

    def step(self, state, action):
        # Upstream mutates Python info/metrics containers. Copy them to preserve
        # predecessor/fork states; device arrays themselves remain immutable.
        state = state.replace(info=dict(state.info), metrics=dict(state.metrics))
        active = state.info["nr_step"] >= self.onset_step
        model = self.mjx_model
        if self.fault_kind == "actuator":
            scales = jnp.ones(12).at[self.leg_map.actuator[self.fault_leg]].set(
                jnp.where(active, self.retained, 1.0))
            model = scale_actuators(model, scales)
        next_state = Joystick.step(_ModelView(self, model), state, action)
        next_step = state.info["nr_step"] + 1
        next_state.info["nr_step"] = next_step
        if self.fault_kind == "sensor":
            reported = next_state.obs["state"][self.sensor_indices]
            frozen = jnp.where(next_step == self.onset_step, reported,
                               state.info["nr_frozen_sensor"])
            active = (next_step >= self.onset_step) & (next_step < self.onset_step + self.freeze_steps)
            actor = next_state.obs["state"].at[self.sensor_indices].set(jnp.where(active, frozen, reported))
            next_state = next_state.replace(obs={**next_state.obs, "state": actor})
            next_state.info["nr_frozen_sensor"] = frozen
        next_state.info["nr_fault_active"] = active & (self.fault_kind != "none")
        return next_state


def observation_manifest(mapping):
    return {"source_commit": "4057c147714b6ac09b395377f1a1724bbeacc4d3",
            "actor_dimension": 48,
            "global_only": {"local_linear_velocity": [0, 3], "gyro": [3, 6],
                            "gravity": [6, 9], "command": [45, 48]},
            "leg_only": {"q_offset": [9, 21], "dq": [21, 33], "previous_action": [33, 45]},
            "leg_mapping": mapping.manifest(), "privileged_actor_access": False,
            "physical_fault_label_to_actor": False, "cross_leg_normalization": False}



from .models.memory import initial_cache


def current_observations(observation):
    if set(observation) != {'state', 'privileged_state'}:
        raise ValueError('Unexpected Go1 observation keys')
    if observation['state'].shape[-1] != 48:
        raise ValueError('Go1 requires current 48-dimensional state')
    # Keep the current official frame. No stacking, masking or actor privilege.
    return {key: observation[key] for key in ('state', 'privileged_state')}

class Go1Base(FaultGo1):
    def __init__(self):
        super().__init__(task='flat_terrain', impl='jax', fault_kind='none', noise_level=1.0)

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(obs=current_observations(state.obs))

    def step(self, state, action):
        state = super().step(state, action)
        return state.replace(obs=current_observations(state.obs))

class FlyCNSGo1(Go1Base):
    def __init__(self, budget=.75, **kwargs):
        self.final_budget = budget
        super().__init__(**kwargs)

    @property
    def action_size(self):
        return 20

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(obs={**state.obs, 'communication': initial_cache(),
            'route_clock': jnp.zeros((1,), jnp.float32),
            'routing_context': jnp.asarray((self.final_budget, 0.), jnp.float32)})

    def step(self, state, action):
        if action.shape[-1] != 20:
            raise ValueError('Expected eight gates and twelve motors')
        memory, clock, context = (state.obs[k] for k in ('communication', 'route_clock', 'routing_context'))
        physical = state.replace(obs={k: state.obs[k] for k in ('state', 'privileged_state')})
        nxt = super().step(physical, action[8:])
        return nxt.replace(obs={**nxt.obs, 'communication': memory,
                                'route_clock': clock+1, 'routing_context': context})
