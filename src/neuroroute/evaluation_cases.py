"""Immutable, policy-independent evaluation scripts. No simulator imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def keyed_seed(namespace, script, stream, step=0):
    """Order-independent unsigned64 seed: policy sampling must use another namespace."""
    return int.from_bytes(hashlib.sha256(canonical([namespace, int(script), stream, int(step)])).digest()[:8], "big")


def key_words(seed_u64):
    """Two unsigned32 words for a backend random key; never route through float."""
    if not isinstance(seed_u64, int) or not 0 <= seed_u64 < 1 << 64:
        raise ValueError("Expected an unsigned64 integer seed")
    return [seed_u64 >> 32, seed_u64 & 0xffffffff]


def unit_uniform(namespace, script, stream, step=0):
    return (keyed_seed(namespace, script, stream, step) >> 11) / float(1 << 53)


def integer_steps(seconds, dt):
    steps = round(seconds / dt)
    if abs(steps * dt - seconds) > 1e-9 or steps < 0:
        raise ValueError("Duration must be an exact nonnegative multiple of control dt")
    return steps


def command_script(config, namespace, script):
    n = integer_steps(config["episode_s"], config["control_dt_s"])
    hold = integer_steps(config["command_hold_s"], config["control_dt_s"])
    if hold <= 0:
        raise ValueError("Command hold duration must be positive")
    current, result = [0., 0., 0.], []
    for index, step in enumerate(range(0, n, hold)):
        for axis in range(3):
            key = f"command:{axis}"
            value = (2 * unit_uniform(namespace, script, key + ":value", index) - 1) * config["command_amplitude"][axis]
            if unit_uniform(namespace, script, key + ":active", index) >= config["command_nonzero_probability"][axis]:
                value = 0.
            if unit_uniform(namespace, script, key + ":keep", index) >= config["command_keep_previous_probability"]:
                current[axis] = round(value, 10)
        result.append({"step": step, "command": list(current)})
    return result


def build_suite(config, split="test"):
    sources = config.get("source_sha256")
    if not sources:
        raise ValueError("Freeze generator and metrics source hashes in the protocol first")
    for name in ("evaluation_cases.py", "metrics.py"):
        actual = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        if sources.get(name) != actual:
            raise ValueError(f"Protocol source hash mismatch: {name}")
    if split not in {"test", "development"}:
        raise ValueError("Explicit test or development namespace is required")
    namespace = config[f"{split}_namespace"]
    dt, horizon = config["control_dt_s"], integer_steps(config["episode_s"], config["control_dt_s"])
    lo, hi = [integer_steps(t, dt) for t in config["fault_onset_s"]]
    if not 0 <= lo <= hi < horizon:
        raise ValueError("Fault onset interval must lie inside the episode")
    cases = []

    def add(kind, script, leg=None, value=None):
        suffix = "" if leg is None else f":{leg}:{value}"
        onset = lo + keyed_seed(namespace, script, "fault_onset") % (hi-lo+1)
        case = {"case_id": f"{split}:{kind}:{script:03d}{suffix}", "kind": kind,
            "script_id": script, "namespace": namespace, "dt_s": dt, "horizon_steps": horizon,
            "command_script": command_script(config, namespace, script),
            "reset_seed_u64": keyed_seed(namespace, script, "reset"),
            "noise_seed_u64": keyed_seed(namespace, script, "observation_noise"),
            "noise_rule": "keyed_seed(namespace,script_id,'observation_noise',absolute_control_step)",
            "terrain_seed_u64": keyed_seed(namespace, script, "terrain") if kind == "rough" else None,
            "fault": None}
        if leg is not None:
            case["fault"] = {"leg": leg, "leg_index": config["legs"].index(leg), "onset_step": onset,
                "onset_s": onset * dt, "loss_fraction": value if kind == "actuator" else None,
                "duration_steps": integer_steps(value, dt) if kind == "sensor_freeze" else horizon-onset}
        case["case_sha256"] = digest(case)
        cases.append(case)

    for kind in ["healthy", "rough"]:
        for script in range(config["counts"][kind]):
            add(kind, script)
    for kind, values in [("actuator", config["actuator_loss_fraction"]), ("sensor_freeze", config["sensor_freeze_s"])]:
        for leg in config["legs"]:
            for value in values:
                for script in range(config["scripts_per_fault_cell"]):
                    add(kind, script, leg, value)
    if dict(Counter(c["kind"] for c in cases)) != config["counts"]:
        raise ValueError("Case matrix does not match declared fixed counts")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case ID")
    return {"split": split, "namespace": namespace, "protocol_sha256": digest(config),
            "source_sha256": sources,
            "cases_sha256": digest(cases), "cases": cases,
            "status": "generated_not_robot_evaluated"}


def write_immutable(path, suite):
    """Never replace a previously frozen suite with different cases."""
    payload = canonical(suite) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen evaluation suite differs: {path}") from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation_protocol.json"))
    parser.add_argument("--split", choices=["test", "development"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = build_suite(json.loads(args.config.read_text()), args.split)
    write_immutable(args.output, suite)
    print(json.dumps({k: suite[k] for k in ["split", "protocol_sha256", "cases_sha256", "status"]}))


if __name__ == "__main__":
    main()
