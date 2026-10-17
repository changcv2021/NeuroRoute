"""New confirmation E3 cases; never relabel or reuse development case keys."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from .evaluation_cases import canonical, digest, key_words

NAMESPACE = "neuroroute.confirmation.e3.v1"
BRANCHES = ("intact", "delay_up", "delay_down", "random_delay", "payload_permutation")
METHOD_BUDGETS = {"B_FULL": 1.0, "B_ASYM": 0.25, "M_BIO": 0.25}
SEEDS = (11, 22, 33, 44, 55)


def keyed_seed(namespace, script, stream, step=0):
    return int.from_bytes(hashlib.sha256(canonical([namespace, str(script), stream, int(step)])).digest()[:8], "big")


@dataclass(frozen=True)
class ConfirmationE3Design:
    phase: str = "confirmation"
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


def load_protocol(path, expected_sha256, *, source_root=None):
    from .runtime import file_sha256
    if file_sha256(path) != expected_sha256:
        raise ValueError("Frozen confirmation E3 protocol SHA mismatch")
    protocol = json.loads(Path(path).read_text())
    expected = {"schema_version": 1, "phase": "confirmation", "status": "frozen",
                "namespace": NAMESPACE, "design": asdict(ConfirmationE3Design()),
                "branches": list(BRANCHES), "events": ["command", "push"],
                "training_seeds": list(SEEDS), "method_budgets": METHOD_BUDGETS,
                "checkpoint_requirement": "complete_version3_confirmation_matching_supplied_training_protocol",
                "fork_execution": "serial_lax_map_shared_scalar_compiled_body",
                "donor_rule": "pre_fork_actual_sent_same_checkpoint_direction_leg_codec_exact_current_command_last_five_permuted",
                "replacement_of_unavailable_forks": False,
                "clean_content_causal_claim_allowed": False,
                "common_training_steps": 200015872, "shards_per_checkpoint": 8}
    for key, value in expected.items():
        if canonical(protocol.get(key)) != canonical(value):
            raise ValueError(f"Confirmation E3 design is not the predeclared non-smoke protocol: {key}")
    root = Path(__file__).parent if source_root is None else Path(source_root)
    guarded = ("e3_evaluation.py", "e3_policy.py", "e3_transport.py", "e3_push.py", "e3_cases.py",
               "policy.py", "communication.py", "environment.py", "training.py", "reproducible_training.py",
               "fast_reproducible_training.py", "confirmation_training.py", "confirmation_protocol.py",
               "evaluation_cases.py", "metrics.py", "development_evaluation.py", "confirmation_evaluation.py",
               "e3_confirmation_cases.py", "e3_confirmation_evaluation.py")
    recorded = protocol.get("frozen_helper_sha256", {})
    if set(recorded) != set(guarded) or any(file_sha256(root/name) != recorded[name] for name in guarded):
        raise ValueError("Frozen E3/core/metrics/runtime helper source mismatch")
    return protocol, ConfirmationE3Design()


def build_cases(protocol_sha256, *, phase="confirmation", smoke=False):
    if phase not in {"confirmation", "validation"} or (phase == "confirmation" and smoke):
        raise ValueError("Only technical validation may use a smoke subset")
    design = ConfirmationE3Design() if phase == "confirmation" else replace(ConfirmationE3Design(), phase=phase)
    if smoke:
        design = replace(design, forks_per_event=2, prefix_steps=24, horizon_steps=32, event_step=2)
    namespace = NAMESPACE if phase == "confirmation" else "neuroroute.validation.e3.v1"
    cases = []
    for event in ("command", "push"):
        for fork in range(design.forks_per_event):
            script = f"{phase}_fork_{fork:03d}"
            case = {"case_id": f"e3_{phase}_{event}_{fork:03d}", "phase": phase,
                    "formal_result": phase == "confirmation", "namespace": namespace, "script_id": script,
                    "event": event, "fork_index": fork, "fork_step": design.prefix_steps,
                    "event_step": design.event_step, "control_direction": "up" if fork % 2 == 0 else "down",
                    "reset_key": key_words(keyed_seed(namespace, script, "reset")),
                    "random_window_seed": keyed_seed(namespace, script, f"{event}:random_window"),
                    "payload_seed": keyed_seed(namespace, script, f"{event}:payload_permutation"),
                    "allowed_random_onsets": list(range(design.event_step+design.window_steps,
                        design.horizon_steps-2*design.window_steps+1, design.random_stride_steps)),
                    "protocol_sha256": protocol_sha256, "design": asdict(design)}
            case["case_sha256"] = digest(case)
            cases.append(case)
    return {"schema_version": 1, "phase": phase, "formal_result": phase == "confirmation",
            "namespace": namespace, "design": asdict(design), "branches": list(BRANCHES), "cases": cases,
            "replacement_rule": "never replace a failed or unavailable predetermined fork"}


def dense_scripts(case, config, method, training_seed):
    import numpy as np
    phase = case["phase"]
    namespace = NAMESPACE if phase == "confirmation" else "neuroroute.validation.e3.v1"
    seeds = SEEDS if phase == "confirmation" else (907001, 907002)
    if phase not in {"confirmation", "validation"} or case["namespace"] != namespace or training_seed not in seeds:
        raise ValueError("E3 scripts and training seeds must belong to the same explicit phase")
    n = config.prefix_steps+config.horizon_steps
    commands = np.tile(np.asarray(config.initial_command, np.float32), (n+1, 1))
    commands[:config.prefix_steps//2] = config.changed_command
    if case["event"] == "command":
        commands[config.prefix_steps+config.event_step:] = config.changed_command
    script = case["script_id"]
    policy_namespace = f"{namespace}.policy:{method}:{training_seed}"
    return {"command": commands[:-1], "next_command": commands[1:],
            "environment_key": np.asarray([key_words(keyed_seed(namespace, script, "observation_noise", t)) for t in range(n)], np.uint32),
            "policy_key": np.asarray([key_words(keyed_seed(policy_namespace, script, "communication", t)) for t in range(n)], np.uint32),
            "absolute_step": np.arange(n, dtype=np.int32),
            "push_enabled": np.full(n, case["event"] == "push", bool)}


def donor_plan(prefix, case, design, checkpoint_sha256):
    import numpy as np
    from .e3_transport import PayloadSemantics, permute_matched_payloads
    direction = case["control_direction"]
    command = np.asarray(design.changed_command if case["event"] == "command" else design.initial_command, np.float32)
    eligible = np.all(np.asarray(prefix["command"]) == command, axis=-1) & np.asarray(prefix["active_before_step"], bool)
    sent = np.asarray(prefix[f"{direction}_sent"], bool)
    payload = np.asarray(prefix[f"{direction}_sent_payload"], np.uint8)
    if len(sent) != design.prefix_steps:
        raise ValueError("Donor input must contain only the completed pre-fork prefix")
    replacement = np.zeros((5, 4, 8), np.uint8)
    support = "flat|exact_command=" + ",".join(float(x).hex() for x in command)
    details = []
    for leg in range(4):
        source = np.flatnonzero(eligible & sent[:, leg])
        if len(source) < 5:
            return None, {"status": "unavailable", "reason": "insufficient_causal_same_command_donors",
                          "direction": direction, "leg": leg, "available_packets": int(len(source)), "required_packets": 5}
        pool = source[-5:]
        semantics = PayloadSemantics(checkpoint_sha256, direction, leg,
                                     "HybridPolicy:uint8[8]:unchanged_symbol_positions", support)
        replacement[:, leg], permutation = permute_matched_payloads(payload[pool, leg], [semantics]*5,
            keyed_seed(case["namespace"], case["case_id"], f"donor_leg:{leg}:{case['payload_seed']}"), sent_flags=sent[pool, leg])
        details.append({"leg": leg, "pool_send_steps": pool.tolist(),
                        "donor_send_steps_in_use_order": pool[permutation].tolist(),
                        "permutation_indices": permutation.tolist(), "support_stratum": support})
    return replacement, {"status": "available_requires_actual_audit", "direction": direction,
        "per_leg": details, "all_donors_strictly_before_fork": True,
        "control_kind": "causal_pre_fork_matched_donor_pool_permutation",
        "target_packet_multiset_preserved": False, "exchangeability_claim": False,
        "limitation": "Matches current command/domain/leg/codec, not every private-state variable; no clean content-causality claim."}


def branch_controls(reference, prefix, case, design, checkpoint_sha256):
    import numpy as np
    from .e3_evaluation import empty_controls
    from .e3_transport import choose_random_matched_window
    plans = [empty_controls(design.horizon_steps) for _ in BRANCHES]
    start, end = design.event_step, design.event_step+design.window_steps
    plans[1]["up_delay"][start:end] = True
    plans[2]["down_delay"][start:end] = True
    direction = case["control_direction"]
    random = choose_random_matched_window(reference[f"{direction}_sent"], start,
        case["allowed_random_onsets"], case["random_window_seed"], window_steps=5)
    if random["onset"] is not None:
        plans[3][f"{direction}_delay"][random["onset"]:random["onset"]+5] = True
    donor, audit = donor_plan(prefix, case, design, checkpoint_sha256)
    if donor is not None:
        plans[4][f"{direction}_replace"][start:end] = True
        plans[4][f"{direction}_replacement"][start:end] = donor
    return {k: np.stack([p[k] for p in plans]) for k in plans[0]}, [True, True, True, random["onset"] is not None, donor is not None], {
        "random_delay": random, "payload_permutation": audit}


def select_shard(suite, shard_index, *, smoke=False):
    count = 1 if smoke else 8
    if type(shard_index) is not int or not 0 <= shard_index < count:
        raise ValueError("E3 shard index must belong to the fixed complete partition")
    cases = suite["cases"]
    if len(cases) != (4 if smoke else 80):
        raise ValueError("Unexpected E3 suite size")
    return {**suite, "suite_cases_sha256": digest(cases), "suite_case_count": len(cases),
            "shard_index": shard_index, "shard_count": count,
            "cases": cases[shard_index::count]}
