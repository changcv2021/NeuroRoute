"""Source-locked OpenAI MPE physics for MATCH paper-structure checks.

This adapter is not claimed to be MATCH's unavailable environment fork.
It preserves original simple_spread dynamics, action convention and shared
reward, including the original self-collision reward offset. N=5/6 is from
MATCH; episode length and entity encoding are declared engineering choices.
"""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


MPE_COMMIT = "83ba4d1aeb00282f7c4acd6912435b3ca642c227"
MPE_CORE_SHA256 = "f55afa1f866ce930abce1509bf82636a45bf5721112a042af8dd466529c4025f"


def load_core():
    path = Path(__file__).resolve().parents[2] / "third_party/openai_mpe/core.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != MPE_CORE_SHA256:
        raise ValueError("Source-locked official MPE core checksum mismatch")
    spec = importlib.util.spec_from_file_location("neuroroute_reference_mpe_core", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


class CooperativeNavigation:
    """NumPy physics; no Gym, renderer, MuJoCo or simulation-pose actor input."""

    def __init__(self, num_agents=5, episode_length=25):
        if num_agents not in (3, 5, 6) or episode_length <= 0:
            raise ValueError("Use original N=3 sanity or MATCH N=5/6 with positive horizon")
        core = load_core()
        self.num_agents = num_agents
        self.num_entities = 2 * num_agents
        self.episode_length = episode_length
        self.world = core.World()
        self.world.dim_c = 2
        self.world.collaborative = True
        self.world.agents = [core.Agent() for _ in range(num_agents)]
        for index, agent in enumerate(self.world.agents):
            agent.name = f"agent {index}"
            agent.collide, agent.silent, agent.size = True, True, 0.15
        self.world.landmarks = [core.Landmark() for _ in range(num_agents)]
        for index, landmark in enumerate(self.world.landmarks):
            landmark.name = f"landmark {index}"
            landmark.collide, landmark.movable = False, False

    def reset(self, seed):
        # Same MT19937 draws/order as original np.random.seed/reset_world;
        # isolate generator ownership instead of modifying process-global RNG.
        rng = np.random.RandomState(seed)
        for agent in self.world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = rng.uniform(-1, 1, 2)
            agent.state.p_vel = np.zeros(2)
            agent.state.c = np.zeros(2)
        for landmark in self.world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = rng.uniform(-1, 1, 2)
            landmark.state.p_vel = np.zeros(2)
        self.t = 0
        return self.observe()

    def observe(self):
        entities = self.world.agents + self.world.landmarks
        pos = np.stack([entity.state.p_pos for entity in entities])
        vel = np.stack([entity.state.p_vel for entity in entities])
        is_agent = np.arange(self.num_entities) < self.num_agents
        relative = pos[None, :, :] - pos[:self.num_agents, None, :]
        identity = np.eye(self.num_agents, self.num_entities, dtype=np.float32)
        # Original agent knows own velocity, but not teammates' velocities.
        own_velocity = vel[None, :, :] * identity[..., None]
        obs = np.concatenate((relative,
            np.broadcast_to(is_agent[None, :, None], (self.num_agents, self.num_entities, 1)),
            identity[..., None], own_velocity), axis=-1).astype(np.float32)
        targets = np.concatenate((pos, vel, is_agent[:, None]), axis=-1).astype(np.float32)
        return {"entities": obs, "visible": np.ones(obs.shape[:-1], dtype=bool), "targets": targets}

    def reward(self):
        # Official Scenario.reward for each agent, then MultiAgentEnv's shared
        # sum, exactly once. Self-collision is intentionally not 'fixed'.
        positions = np.stack([a.state.p_pos for a in self.world.agents])
        landmarks = np.stack([l.state.p_pos for l in self.world.landmarks])
        coverage = np.linalg.norm(positions[:, None] - landmarks[None], axis=-1).min(axis=0).sum()
        sizes = np.array([a.size for a in self.world.agents])
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        collisions = distances < sizes[:, None] + sizes[None]
        return float(-self.num_agents * coverage - collisions.sum())

    def percentage_occupied(self):
        agents = np.stack([a.state.p_pos for a in self.world.agents])
        landmarks = np.stack([l.state.p_pos for l in self.world.landmarks])
        nearest = np.linalg.norm(agents[:, None] - landmarks[None], axis=-1).min(axis=0)
        return float((nearest <= 0.3).mean())

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.num_agents,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("Expected one integer action per agent")
        if np.any((actions < 0) | (actions > 4)) or self.t >= self.episode_length:
            raise ValueError("Invalid action or step after finite-horizon episode")
        # Original default accepts one-hots: action1 gives +x, unlike its
        # optional discrete_action_input branch, which reverses these signs.
        directions = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        for agent, action in zip(self.world.agents, actions):
            sensitivity = 5.0 if agent.accel is None else agent.accel
            agent.action.u = sensitivity * directions[action]
            agent.action.c = np.zeros(self.world.dim_c)
        self.world.step()  # Execute unmodified, checksummed OpenAI physics.
        self.t += 1
        return self.observe(), self.reward(), self.t == self.episode_length
