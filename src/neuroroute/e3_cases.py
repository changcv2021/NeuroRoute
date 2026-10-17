"""Predetermined development E3 cases and causal, typed packet donor plans.

This module deliberately imports no simulator/JAX. Case construction must finish
before checkpoint execution. No outcome-dependent fork replacement is supported.
"""

from dataclasses import asdict, dataclass
import hashlib
import math

from .evaluation_cases import canonical, digest, key_words


BRANCHES = ("intact", "delay_up", "delay_down", "random_delay", "payload_permutation")
NAMESPACE = "neuroroute.development.e3.v1"


def keyed_seed(namespace, script, stream, step=0):
    """E3 uses string fork IDs; standard evaluation retains its integer IDs."""
    return int.from_bytes(hashlib.sha256(
        canonical([namespace, str(script), stream, int(step)])).digest()[:8], "big")


@dataclass(frozen=True)
class E3Config:
    phase: str = "development"
    forks_per_event: int = 40
    dt_s: float = 0.02
    prefix_steps: int = 200
    horizon_steps: int = 200
    event_step: int = 10
    window_steps: int = 5
    initial_command: tuple = (0.3, 0.0, 0.0)
    changed_command: tuple = (0.6, 0.0, 0.0)
    push_force_world_n: tuple = (20.0, 0.0, 0.0)
    push_duration_steps: int = 5
    random_stride_steps: int = 10
    recovery_hold_s: float = 0.5
    recovery_threshold: float = 0.8
    score_sigma_v: float = 0.5
    score_sigma_w: float = 0.5

    def validate(self):
        if self.phase != "development":
            raise ValueError("E3 v1 is a technical development runner; formal model selection is unresolved")
        integer_fields = (self.forks_per_event, self.prefix_steps, self.horizon_steps,
                          self.event_step, self.window_steps, self.push_duration_steps,
                          self.random_stride_steps)
        if any(type(v) is not int for v in integer_fields):
            raise ValueError("Step/count fields must be actual integers")
        if not 1 <= self.forks_per_event <= 40 or self.prefix_steps < 2:
            raise ValueError("Use 1..40 predetermined forks per event and a positive prefix")
        if self.dt_s != 0.02 or self.window_steps != 5:
            raise ValueError("This E3 transport requires exactly 100 ms / five 20 ms steps")
        if (self.event_step < 0 or self.push_duration_steps <= 0 or self.random_stride_steps <= 0
                or self.horizon_steps < self.event_step + max(2*self.window_steps, self.push_duration_steps) + 1):
            raise ValueError("Horizon must include the event and a complete queue-clear tail")
        for vector in (self.initial_command, self.changed_command, self.push_force_world_n):
            if len(vector) != 3 or not all(math.isfinite(x) for x in vector):
                raise ValueError("Commands and force must be finite XYZ vectors")
        if tuple(self.initial_command) == tuple(self.changed_command):
            raise ValueError("The command event must actually change the command")
        if not any(self.push_force_world_n):
            raise ValueError("The force event must have a nonzero physical force")
        if any(not math.isfinite(x) or x <= 0 for x in (self.recovery_hold_s, self.score_sigma_v, self.score_sigma_w)):
            raise ValueError("Metric scales must be positive")
        hold = round(self.recovery_hold_s/self.dt_s)
        if hold < 1 or not math.isclose(hold*self.dt_s, self.recovery_hold_s, abs_tol=1e-9):
            raise ValueError("Recovery hold must be an exact number of control steps")
        if not 0 < self.recovery_threshold <= 1:
            raise ValueError("Invalid recovery threshold")
        return self


def build_cases(config):
    """Two events x fixed reset/script IDs; a failed prefix stays unavailable."""
    config.validate()
    result = []
    for event in ("command", "push"):
        for fork in range(config.forks_per_event):
            script = f"fork_{fork:03d}"
            case = {"case_id": f"{event}_{script}", "namespace": NAMESPACE,
                    "script_id": script, "event": event, "fork_index": fork,
                    "fork_step": config.prefix_steps, "event_step": config.event_step,
                    "control_direction": "up" if fork % 2 == 0 else "down",
                    "reset_key": key_words(keyed_seed(NAMESPACE, script, "reset")),
                    "random_window_seed": keyed_seed(NAMESPACE, script, f"{event}:random_window"),
                    "payload_seed": keyed_seed(NAMESPACE, script, f"{event}:payload_permutation"),
                    "allowed_random_onsets": list(range(config.event_step + config.window_steps,
                        config.horizon_steps - 2*config.window_steps + 1, config.random_stride_steps)),
                    "config": asdict(config)}
            case["case_sha256"] = digest(case)
            result.append(case)
    return {"schema_version": 1, "phase": "development", "formal_result": False,
            "namespace": NAMESPACE, "branches": list(BRANCHES), "cases": result,
            "replacement_rule": "never replace a failed or unavailable predetermined fork"}


def dense_scripts(case, config, method, training_seed):
    """Absolute-step exogenous streams shared by all five counterfactuals.

    Warm-up visits the later command *before* the fork, providing causal donor
    packets in both command strata. This fixed command script is policy-blind.
    """
    import numpy as np
    n = config.prefix_steps + config.horizon_steps
    commands = np.tile(np.asarray(config.initial_command, np.float32), (n+1, 1))
    commands[:config.prefix_steps//2] = config.changed_command
    if case["event"] == "command":
        commands[config.prefix_steps+config.event_step:] = config.changed_command
    script = case["script_id"]
    policy_namespace = f"{NAMESPACE}.policy:{method}:{training_seed}"
    # Event kind is intentionally absent from exogenous streams. Paired events
    # reuse the same prefix; changed physics/commands are the actual treatment.
    return {"command": commands[:-1], "next_command": commands[1:],
            "environment_key": np.asarray([key_words(keyed_seed(NAMESPACE, script, "observation_noise", t)) for t in range(n)], np.uint32),
            "policy_key": np.asarray([key_words(keyed_seed(policy_namespace, script, "communication", t)) for t in range(n)], np.uint32),
            "absolute_step": np.arange(n, dtype=np.int32),
            "push_enabled": np.full(n, case["event"] == "push", bool)}


def causal_donor_plan(prefix_trace, case, config, checkpoint_sha256):
    """Permute pre-fork actually sent packets, preserving direction/leg/codec.

    Donor support is the same flat domain and exact current command. This is
    explicitly a donor-pool permutation intervention, not a claim of conditional
    exchangeability of all private states or a permutation of future messages.
    """
    import numpy as np
    from .e3_transport import PayloadSemantics, permute_matched_payloads
    direction = case["control_direction"]
    command = np.asarray(config.changed_command if case["event"] == "command" else config.initial_command, np.float32)
    in_stratum = np.all(np.asarray(prefix_trace["command"]) == command, axis=-1)
    in_stratum &= np.asarray(prefix_trace["active_before_step"], bool)
    sent = np.asarray(prefix_trace[f"{direction}_sent"], bool)
    payload = np.asarray(prefix_trace[f"{direction}_sent_payload"], np.uint8)
    result = np.zeros((config.window_steps, 4, 8), np.uint8)
    donor_steps = np.full((config.window_steps, 4), -1, np.int32)
    details = []
    support = "flat|exact_command=" + ",".join(float(x).hex() for x in command)
    for leg in range(4):
        candidates = np.flatnonzero(in_stratum & sent[:, leg])
        if len(candidates) < config.window_steps:
            return None, {"status": "unavailable", "reason": "insufficient_causal_same_command_donors",
                          "direction": direction, "leg": leg, "available_packets": len(candidates),
                          "required_packets": config.window_steps}
        selected = candidates[-config.window_steps:]
        semantics = PayloadSemantics(checkpoint_sha256, direction, leg,
                                     "HybridPolicy:uint8[8]:unchanged_symbol_positions", support)
        permuted, indices = permute_matched_payloads(payload[selected, leg], [semantics]*len(selected),
            keyed_seed(NAMESPACE, case["case_id"], f"donor_leg:{leg}:{case['payload_seed']}"),
            sent_flags=sent[selected, leg])
        result[:, leg] = permuted
        donor_steps[:, leg] = selected[indices]
        details.append({"leg": leg, "pool_send_steps": selected.tolist(),
                        "donor_send_steps_in_use_order": selected[indices].tolist(),
                        "permutation_indices": indices.tolist(), "support_stratum": support})
    return result, {"status": "available_requires_actual_audit", "direction": direction,
        "donor_send_steps": donor_steps.tolist(), "per_leg": details,
        "all_donors_strictly_before_fork": bool(np.all(donor_steps < config.prefix_steps)),
        "control_kind": "causal_pre_fork_matched_donor_pool_permutation",
        "target_packet_multiset_preserved": False,
        "exchangeability_claim": False,
        "limitation": "Matches command/domain/leg/codec, not every private-state variable; no clean content-causality claim."}
