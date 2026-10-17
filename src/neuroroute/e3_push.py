"""Real Cartesian wrench events for E3; no observation/fault-label injection."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ForceEvent:
    body_name: str
    onset_step: int
    duration_steps: int
    force_world_n: tuple
    torque_world_nm: tuple = (0.0, 0.0, 0.0)

    def resolve(self, model, control_dt):
        if self.body_name != "trunk":
            raise ValueError("E3 v1 applies torso pushes only to the audited 'trunk' body")
        body_id = int(model.body(self.body_name).id)
        if body_id <= 0:
            raise ValueError("Push body must be a real moving torso, not the world")
        if (self.onset_step < 0 or self.duration_steps <= 0
                or len(self.force_world_n) != 3 or len(self.torque_world_nm) != 3
                or not np.isfinite(tuple(self.force_world_n) + tuple(self.torque_world_nm)).all()
                or not np.isfinite(control_dt) or control_dt <= 0):
            raise ValueError("Invalid force event")
        return ResolvedForceEvent(body_id, self.body_name, self.onset_step, self.duration_steps,
                                  self.force_world_n, self.torque_world_nm, float(control_dt))


@dataclass(frozen=True)
class ResolvedForceEvent:
    body_id: int
    body_name: str
    onset_step: int
    duration_steps: int
    force_world_n: tuple
    torque_world_nm: tuple
    control_dt: float

    def manifest(self):
        duration = self.duration_steps * self.control_dt
        return {"body_id": self.body_id, "body_name": self.body_name,
                "frame": "world", "application_point": "body_center_of_mass",
                "force_N": list(self.force_world_n), "torque_Nm": list(self.torque_world_nm),
                "onset_step_inclusive": self.onset_step,
                "end_step_exclusive": self.onset_step + self.duration_steps,
                "duration_s": duration,
                "impulse_Ns": (np.asarray(self.force_world_n) * duration).tolist(),
                "angular_impulse_Nms": (np.asarray(self.torque_world_nm) * duration).tolist(),
                "mechanism": "data.xfrc_applied, force_xyz followed by torque_xyz",
                "event_metadata_visible_to_actor": False}


def empty_push_state(batch_shape=()):
    """Track only this event's currently applied wrench so it can be removed."""
    return jnp.zeros(tuple(batch_shape) + (6,), jnp.float32)


def apply_force_event(data, event, step, previous_owned_wrench):
    """Set event force for the upcoming control interval, preserving other forces.

    Pass returned owned wrench into the next call. Calling at or after the
    exclusive end removes this event's force; it is not allowed to persist.
    Official random perturbations must be disabled when using this owner model,
    as their independent whole-array replacement would require composition.
    """
    step = jnp.asarray(step, jnp.int32)
    active = (step >= event.onset_step) & (step < event.onset_step + event.duration_steps)
    wrench = jnp.asarray(tuple(event.force_world_n) + tuple(event.torque_world_nm), jnp.float32)
    owned = jnp.where(active[..., None], wrench, jnp.zeros_like(wrench))
    owned = jnp.broadcast_to(owned, previous_owned_wrench.shape)
    base = data.xfrc_applied[..., event.body_id, :] - previous_owned_wrench
    forces = data.xfrc_applied.at[..., event.body_id, :].set(base + owned)
    return data.replace(xfrc_applied=forces), owned


def clear_force_event(data, event, previous_owned_wrench):
    return apply_force_event(data, event, event.onset_step + event.duration_steps,
                              previous_owned_wrench)
