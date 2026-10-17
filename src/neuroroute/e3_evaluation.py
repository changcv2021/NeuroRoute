"""Isolated development E3 forks. No training or checkpoint mutation occurs.

Five branches start from one materialized physical/history/cache/queue snapshot.
The intact reference is a genuine producer for random-count matching; donors
are restricted to the already completed prefix. All RNG is keyed by absolute
step, independently of actions, termination and branch identity.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time

from .e3_cases import BRANCHES, E3Config, build_cases, dense_scripts, causal_donor_plan
from .runtime import require_compute, write_json, file_sha256


def make_rollout(env, actor, config, experiment, *, fork_execution="serial_map"):
    import jax
    import jax.numpy as jnp
    from .communication import empty_state, full_gates, periodic_gates
    from .e3_transport import empty_transport, transmission_cost
    from .e3_push import ForceEvent, apply_force_event, clear_force_event, empty_push_state
    from .environment import assemble_actions, split_observations
    from .training import initialize_history, append_history, select_environments
    from .reproducible_training import canonical_arrays

    event = ForceEvent("trunk", experiment.prefix_steps + experiment.event_step,
                      experiment.push_duration_steps, tuple(experiment.push_force_world_n)).resolve(env.mj_model, env.dt)
    command_counter = experiment.prefix_steps + experiment.horizon_steps + 1

    def override_command(state, command):
        info, obs = dict(state.info), dict(state.obs)
        info["command"] = command
        info["steps_until_next_cmd"] = jnp.asarray(command_counter, jnp.int32)
        obs["state"] = obs["state"].at[45:48].set(command)
        obs["privileged_state"] = obs["privileged_state"].at[45:48].set(command)
        return state.replace(info=info, obs=obs)

    def reset(key, command):
        # Official reset includes Python scalar leaves before its first JIT boundary.
        state = canonical_arrays(jax.tree.map(jnp.asarray, override_command(env.reset(key), command)))
        history = initialize_history(*split_observations(state.obs, env.leg_map), config.history_length)
        return (state, history, empty_state(), empty_transport(), empty_push_state(), jnp.asarray(True, jnp.bool_))

    def rollout(snapshot, params, arrays, controls):
        def step_fn(carry, inputs):
            state, history, cache, transport, owned, live = carry
            a, c = inputs
            if config.method == "B_FULL":
                forced = full_gates()
            elif config.method == "B_PERIODIC":
                forced = periodic_gates(a["absolute_step"], config.budget_fraction)
            else:
                forced = None
            g, l = history.actor_inputs()
            output = actor.apply({"params": params}, g, l, cache, transport, a["policy_key"],
                forced_gates=forced, deterministic_motor=True,
                up_delay_mask=c["up_delay"], down_delay_mask=c["down_delay"],
                up_replacement=c["up_replacement"], down_replacement=c["down_replacement"],
                up_replace_mask=c["up_replace"], down_replace_mask=c["down_replace"])
            state = override_command(state, a["command"])
            state = state.replace(info={**state.info, "rng": a["environment_key"]}, metrics=dict(state.metrics))
            pushed, next_owned = apply_force_event(state.data, event, a["absolute_step"], owned)
            next_owned = jnp.where(a["push_enabled"], next_owned, jnp.zeros_like(next_owned))
            pushed = select_environments(a["push_enabled"], pushed, state.data)
            nxt = canonical_arrays(env.step(state.replace(data=pushed), assemble_actions(output.policy.motor_action, env.leg_map)))
            nxt = canonical_arrays(override_command(nxt, a["next_command"]))
            velocity, yaw = env.get_local_linvel(nxt.data)[:2], env.get_gyro(nxt.data)[2]
            finite = (jnp.all(jnp.isfinite(nxt.data.qpos)) & jnp.all(jnp.isfinite(nxt.data.qvel))
                      & jnp.all(jnp.isfinite(nxt.obs["state"])) & jnp.all(jnp.isfinite(output.policy.motor_action))
                      & jnp.all(jnp.isfinite(velocity)) & jnp.isfinite(yaw) & jnp.isfinite(nxt.reward))
            ended = nxt.done.astype(jnp.bool_) | ~finite
            live_next = live & ~ended
            next_history = append_history(history, *split_observations(nxt.obs, env.leg_map))
            # The force acted for the recorded transition. A stopped trajectory
            # relinquishes its force owner immediately; no residual applied force.
            cleared, cleared_owned = clear_force_event(nxt.data, event, next_owned)
            nxt = nxt.replace(data=select_environments(ended, cleared, nxt.data))
            retained_owned = jnp.where(ended, cleared_owned, next_owned)
            record = {"active_before_step": live, "terminated_this_step": live & ended,
                "nonfinite_this_step": live & ~finite, "absolute_step": a["absolute_step"],
                "command": a["command"], "velocity_xy": jnp.where(live, velocity, 0), "yaw_rate": jnp.where(live, yaw, 0),
                "qpos": jnp.where(live, nxt.data.qpos, state.data.qpos), "qvel": jnp.where(live, nxt.data.qvel, state.data.qvel),
                "actuator_force": jnp.where(live, nxt.data.actuator_force, 0), "motor_action": jnp.where(live, output.policy.motor_action, 0),
                "raw_reward": jnp.where(live, nxt.reward, 0),
                "owned_wrench_during_transition": jnp.where(live, next_owned, 0),
                "owned_wrench_after_transition": jnp.where(live, retained_owned, 0),
                "event_applied_before_transition": live & (a["absolute_step"] >= experiment.prefix_steps + experiment.event_step),
                "actor_command_error": jnp.where(live, jnp.max(jnp.abs(history.global_obs[-1, -3:] - a["command"])), 0),
                "policy_key": a["policy_key"], "environment_key": a["environment_key"],
                "cache_age_s": jnp.where(live, jnp.stack((output.policy.cache.up.age, output.policy.cache.down.age)), jnp.stack((cache.up.age, cache.down.age))),
                "cache_valid": jnp.where(live, jnp.stack((output.policy.cache.up.valid, output.policy.cache.down.valid)), jnp.stack((cache.up.valid, cache.down.valid))),
                "cache_payload": jnp.where(live, jnp.stack((output.policy.cache.up.payload, output.policy.cache.down.payload)), jnp.stack((cache.up.payload, cache.down.payload))),
                "sent_total_bits": jnp.where(live, transmission_cost(output.up_log, output.down_log)["total_bits"], 0)}
            for direction, log in (("up", output.up_log), ("down", output.down_log)):
                for name in log.__dataclass_fields__:
                    value = getattr(log, name)
                    # Post-fall slots are padding, never packet sends/receipts.
                    if name in {"sent", "selected_for_delay", "content_replaced", "content_changed", "arrived", "cache_updated", "overflow"}:
                        value = value & live
                    else:
                        padding = -1 if name in {"send_step", "scheduled_arrival_step", "arrived_send_step", "arrival_step", "cache_source_send_step"} else 0
                        value = jnp.where(live, value, jnp.asarray(padding, value.dtype))
                    record[f"{direction}_{name}"] = value
            candidate = (nxt, next_history, output.policy.cache, output.transport, retained_owned, live_next)
            return select_environments(live, candidate, carry), record
        return jax.lax.scan(step_fn, snapshot, (arrays, controls))

    if fork_execution == "serial_map":
        # A vmap lane is not a numerical identity: GPU reduction order can
        # differ by lane even for equal inputs. Each causal branch therefore
        # executes the very same scalar body and scalar physical array shapes.
        # Independent cases/checkpoints can still occupy separate GPU jobs.
        def grouped(snapshots, params, arrays, controls):
            return jax.lax.map(lambda pair: rollout(pair[0], params, arrays, pair[1]),
                               (snapshots, controls))
    elif fork_execution == "vmap_diagnostic_only":
        grouped = jax.vmap(rollout, in_axes=(0, None, None, 0))
    else:
        raise ValueError("Unsupported E3 fork execution mode")
    return jax.jit(reset), jax.jit(rollout), jax.jit(grouped), event


def empty_controls(n):
    import numpy as np
    return {"up_delay": np.zeros((n, 4), bool), "down_delay": np.zeros((n, 4), bool),
            "up_replacement": np.zeros((n, 4, 8), np.uint8), "down_replacement": np.zeros((n, 4, 8), np.uint8),
            "up_replace": np.zeros((n, 4), bool), "down_replace": np.zeros((n, 4), bool)}


def branch_controls(reference, prefix, case, experiment, checkpoint_sha256):
    import numpy as np
    from .e3_transport import choose_random_matched_window
    plans = [empty_controls(experiment.horizon_steps) for _ in BRANCHES]
    start, end = experiment.event_step, experiment.event_step + experiment.window_steps
    plans[1]["up_delay"][start:end] = True
    plans[2]["down_delay"][start:end] = True
    direction = case["control_direction"]
    random = choose_random_matched_window(reference[f"{direction}_sent"], start,
        case["allowed_random_onsets"], case["random_window_seed"], window_steps=experiment.window_steps)
    if random["onset"] is not None:
        plans[3][f"{direction}_delay"][random["onset"]:random["onset"]+experiment.window_steps] = True
    donor, donor_audit = causal_donor_plan(prefix, case, experiment, checkpoint_sha256)
    if donor is not None:
        plans[4][f"{direction}_replace"][start:end] = True
        plans[4][f"{direction}_replacement"][start:end] = donor
    availability = [True, True, True, random["onset"] is not None, donor is not None]
    return {k: np.stack([p[k] for p in plans]) for k in plans[0]}, availability, {"random_delay": random, "payload_permutation": donor_audit}


def tree_comparison(first, second, *, atol=2e-5):
    """Every numerical state leaf; integers/bools/keys must be exactly equal."""
    import jax
    import numpy as np
    ap, at = jax.tree_util.tree_flatten_with_path(first)
    bp, bt = jax.tree_util.tree_flatten_with_path(second)
    if at != bt:
        raise AssertionError("Fork/replay state pytree structure differs")
    failures, max_error = [], 0.0
    for (path, a), (_, b) in zip(ap, bp):
        a, b = np.asarray(a), np.asarray(b)
        exact = a.dtype.kind not in "fc"
        okay = (a.shape == b.shape and a.dtype == b.dtype and
                (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=0, atol=atol, equal_nan=False)))
        err = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), initial=0)) if a.shape == b.shape else None
        if err is not None and np.isfinite(err):
            max_error = max(max_error, err)
        if not okay:
            failures.append({"path": jax.tree_util.keystr(path), "max_abs_error": err if err is not None and np.isfinite(err) else None})
    return {"passed": not failures, "leaf_count": len(ap), "atol": atol, "rtol": 0,
            "max_abs_error": max_error, "failures": failures}


def summarize_branch(trace, experiment):
    import numpy as np
    from .metrics import tracking_metrics, recovery_metrics
    from .e3_transport import DeliveryLog, exposure_summary
    failures = np.flatnonzero(trace["terminated_this_step"])
    failure = int(failures[0]) if len(failures) else None
    active_steps = int(np.asarray(trace["active_before_step"]).sum())
    tracking = tracking_metrics(trace["velocity_xy"], trace["yaw_rate"], trace["command"],
        horizon_steps=experiment.horizon_steps, dt_s=experiment.dt_s, failure_step=failure,
        sigma_v=experiment.score_sigma_v, sigma_w=experiment.score_sigma_w)
    scores = tracking.pop("score_trace")
    recovery = recovery_metrics(scores, event_step=experiment.event_step, dt_s=experiment.dt_s,
        failure_step=failure, score_threshold=experiment.recovery_threshold, hold_s=experiment.recovery_hold_s)
    event_applied = bool(np.any(trace["event_applied_before_transition"]))
    recovery["sample_index_failed_before_event"] = recovery["failed_before_event"]
    recovery["event_applied_before_failure"] = event_applied
    if recovery["failed_before_event"] and event_applied:
        recovery.update(failed_before_event=False, status="failed_after_event", censor_time_s=experiment.dt_s)
    exposure = {}
    for direction in ("up", "down"):
        log = DeliveryLog(**{name: trace[f"{direction}_{name}"] for name in DeliveryLog.__dataclass_fields__})
        exposure[direction] = exposure_summary(log, experiment.dt_s)
        if np.any(trace[f"{direction}_overflow"]):
            raise AssertionError("E3 packet queue overflow; abort rather than lose or undercharge packets")
    sent_payload = sum(x["sent_payload_bits"] for x in exposure.values())
    total_bits = int(np.sum(trace["sent_total_bits"]))
    if total_bits != 8*active_steps + sent_payload:
        raise AssertionError("Source send accounting failed")
    if np.any(np.asarray(trace["actor_command_error"]) > 1e-7):
        raise AssertionError("Command script and allowed actor history are misaligned")
    return {"tracking": tracking, "recovery": recovery, "exposure": exposure,
            "sent_flag_bits": 8*active_steps, "sent_total_bits": total_bits,
            "active_steps": active_steps, "active_bits_per_step": total_bits/active_steps if active_steps else None,
            "received_payload_bits": sum(x["received_payload_bits"] for x in exposure.values()),
            "nonfinite_failure": bool(np.any(trace["nonfinite_this_step"])),
            "actual_push_impulse_Ns": (np.asarray(trace["owned_wrench_during_transition"], np.float64)[:, :3].sum(0)*experiment.dt_s).tolist(),
            "pending_receipts_are_censored_at_stop": bool(any(x["pending_payload_bits"] for x in exposure.values())),
            "score_trace": scores}


def run(args):
    require_compute(gpu=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .environment import FaultGo1
    from .training import TrainConfig
    from .policy import HybridPolicy
    from .e3_policy import InterventionPolicy
    from .communication import full_gates
    from .e3_transport import compare_exposure
    from .development_evaluation import mixed_runtime_candidate, runtime_metadata
    from .reproducible_training import ReproduciblePPOTrainer
    from .fast_reproducible_training import FastReproduciblePPOTrainer

    started = time.monotonic()
    experiment = E3Config(**json.loads(args.config.read_text())).validate()
    suite = build_cases(experiment)
    if args.output.exists():
        raise FileExistsError("Use a new E3 output directory; prior branches are immutable")
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint differs from the explicitly frozen producer hash")
    args.output.mkdir(parents=True)
    write_json(args.output/"cases.json", suite)
    write_json(args.output/"configuration.json", asdict(experiment))
    result = {"status": "failed", "phase": "development", "formal_result": False,
              "performance_claim": False, "checkpoint_sha256": args.checkpoint_sha256}
    try:
        # Trusted locally generated checkpoint only. Never deserialize uploads.
        with args.checkpoint.open("rb") as stream:
            header = pickle.load(stream)
        if header.get("format_version") != 2:
            raise ValueError("E3 requires a runtime-bound version2 checkpoint")
        candidate = mixed_runtime_candidate(header)
        config = TrainConfig(**header["config"])
        if config.method not in {"B_FULL", "B_ASYM", "M_BIO"}:
            raise ValueError("E3 development currently covers B_FULL, B_ASYM, M_BIO checkpoints only")
        env = FaultGo1(impl="jax")
        if env._config.pert_config.enable or not np.isclose(env.dt, experiment.dt_s, rtol=0, atol=1e-10):
            raise ValueError("E3 requires audited dt and disabled independent random perturbations")
        validator = (FastReproduciblePPOTrainer if candidate == "jax082_fast_scatter" else ReproduciblePPOTrainer)(env, config)
        loaded = validator.load_checkpoint(args.checkpoint)
        params = loaded.params["actor"]
        actor = InterventionPolicy(config.local_width, config.global_width, control_dt=env.dt)
        original_actor = HybridPolicy(config.local_width, config.global_width)
        reset, prefix_rollout, fork_rollout, event = make_rollout(env, actor, config, experiment)
        provenance = {"checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": args.checkpoint_sha256,
            "checkpoint_format": 2, "numerical_runtime": validator.runtime,
            "checkpoint_update_index": int(loaded.update_index), "checkpoint_env_steps": int(loaded.environment_steps),
            "training_method": config.method, "training_seed": config.seed,
            "environment": runtime_metadata(env), "push": event.manifest(),
            "push_strength_status": "development_diagnostic_20N_default_not_formal_calibration",
            "normalizer": header["normalizer"], "source_hashes": {p.name: file_sha256(p) for p in
                [Path(__file__), Path(__file__).with_name("e3_cases.py"), Path(__file__).with_name("e3_policy.py"),
                 Path(__file__).with_name("e3_transport.py"), Path(__file__).with_name("e3_push.py"), args.config]},
            "job_id": os.environ["SLURM_JOB_ID"], "rng_namespace": suite["namespace"],
            "motor_policy": "conditional_mean_with_sampled_messages_and_gates",
            "fork_execution": "serial_lax_map_shared_scalar_compiled_body",
            "actor_inputs": "only legal 3-frame global/local observations plus received packet cache",
            "formal_model_selection_resolved": False, "clean_content_causal_claim_allowed": False,
            "snapshot_semantics": "full MJX/solver state, environment info RNG/command scheduler, history, received cache, pending queues, next absolute script keys; fixed identity normalizer"}
        write_json(args.output/"provenance.json", provenance)
        summaries = []
        for case in suite["cases"]:
            case_started = time.monotonic()
            directory = args.output/case["case_id"]
            directory.mkdir()
            arrays = dense_scripts(case, experiment, config.method, config.seed)
            n = experiment.prefix_steps
            initial = reset(jnp.asarray(case["reset_key"], jnp.uint32), jnp.asarray(arrays["command"][0]))
            prefix_arrays = {k: jnp.asarray(v[:n]) for k, v in arrays.items()}
            branch_arrays = {k: jnp.asarray(v[n:]) for k, v in arrays.items()}
            snapshot, prefix_trace = prefix_rollout(initial, params, prefix_arrays,
                                                   jax.tree.map(jnp.asarray, empty_controls(n)))
            snapshot, prefix_trace = jax.device_get((snapshot, prefix_trace))
            np.savez_compressed(directory/"prefix_trace.npz", **prefix_trace)
            prefix_status = {"case_id": case["case_id"], "case_sha256": case["case_sha256"],
                             "fork_step": n, "available": bool(snapshot[-1]),
                             "prefix_trace_sha256": file_sha256(directory/"prefix_trace.npz")}
            if not bool(snapshot[-1]):
                summary = {**prefix_status, "status": "unavailable", "reason": "failed_before_predetermined_fork",
                           "first_failure_step": int(np.flatnonzero(prefix_trace["terminated_this_step"])[0]),
                           "branches": {b: {"status": "unavailable"} for b in BRANCHES}}
                write_json(directory/"summary.json", summary)
                summaries.append(summary)
                print(json.dumps({"case_id": case["case_id"], "status": summary["status"], "elapsed_s": time.monotonic()-case_started}), flush=True)
                continue
            # Snapshot is materialized once. All five branches derive from these
            # exact same leaves, never from another reset or a replayed prefix.
            with (directory/"fork_snapshot.pkl").open("wb") as stream:
                pickle.dump({"snapshot": snapshot, "next_script_inputs": jax.device_get(branch_arrays),
                             "case": case, "checkpoint_sha256": args.checkpoint_sha256,
                             "normalizer": header["normalizer"]}, stream, protocol=5)
            snapshot_hash = file_sha256(directory/"fork_snapshot.pkl")
            g, l = snapshot[1].actor_inputs()
            forced = full_gates() if config.method == "B_FULL" else None
            original = original_actor.apply({"params": params}, jnp.asarray(g), jnp.asarray(l),
                jax.tree.map(jnp.asarray, snapshot[2]), branch_arrays["policy_key"][0], env.dt,
                deterministic_motor=True, forced_gates=forced)
            adapted = actor.apply({"params": params}, jnp.asarray(g), jnp.asarray(l),
                jax.tree.map(jnp.asarray, snapshot[2]), jax.tree.map(jnp.asarray, snapshot[3]),
                branch_arrays["policy_key"][0], deterministic_motor=True, forced_gates=forced)
            alignment = tree_comparison(jax.device_get(original), jax.device_get(adapted.policy))
            if not alignment["passed"]:
                write_json(directory/"intact_alignment.json", alignment)
                raise AssertionError("Intervention adapter differs from the trained intact HybridPolicy")
            five_snapshots = jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x), (5,)+np.shape(x)), snapshot)
            no_interventions = jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x), (5,)+np.shape(x)), empty_controls(experiment.horizon_steps))
            reference_final, reference_five = fork_rollout(five_snapshots, params, branch_arrays, no_interventions)
            reference_final, reference_five = jax.device_get((reference_final, reference_five))
            reference = jax.tree.map(lambda x: x[0], reference_five)
            reference_slot_checks = [tree_comparison(
                (jax.tree.map(lambda x: x[0], reference_final), reference),
                (jax.tree.map(lambda x: x[i], reference_final), jax.tree.map(lambda x: x[i], reference_five)))
                for i in range(5)]
            reference_slot_exact = [all(np.array_equal(x[0], x[i]) for x in jax.tree.leaves((reference_final, reference_five))) for i in range(5)]
            write_json(directory/"reference_slot_alignment.json", {
                "passed": all(c["passed"] for c in reference_slot_checks), "slots": reference_slot_checks,
                "arrays_exactly_equal_by_slot": reference_slot_exact,
                "execution": "serial_lax_map_shared_scalar_compiled_body"})
            if not all(c["passed"] for c in reference_slot_checks):
                raise AssertionError("Identical intact inputs differ across fork execution positions")
            np.savez_compressed(directory/"intact_reference.npz", **reference)
            # Controls cannot be constructed before this intact producer exists.
            controls, available, control_audit = branch_controls(reference, prefix_trace, case, experiment, args.checkpoint_sha256)
            np.savez_compressed(directory/"branch_controls.npz", **controls)
            write_json(directory/"control_plan.json", {**control_audit, "available": dict(zip(BRANCHES, available)),
                "producer_reference_sha256": file_sha256(directory/"intact_reference.npz"),
                "donor_prefix_sha256": prefix_status["prefix_trace_sha256"],
                "controls_sha256": file_sha256(directory/"branch_controls.npz")})
            final, traces = fork_rollout(five_snapshots, params, branch_arrays, jax.tree.map(jnp.asarray, controls))
            final, traces = jax.device_get((final, traces))
            replay = tree_comparison((jax.tree.map(lambda x: x[0], reference_final), reference),
                                     (jax.tree.map(lambda x: x[0], final), jax.tree.map(lambda x: x[0], traces)))
            if not replay["passed"]:
                write_json(directory/"intact_replay_failure.json", replay)
                raise AssertionError("Same intact branch state/trace failed fixed 2e-5 replay tolerance")
            branches = {}
            for i, name in enumerate(BRANCHES):
                if not available[i]:
                    branches[name] = {"status": "unavailable", "reason": control_audit[name]["reason"] if "reason" in control_audit[name] else "no_reference_count_matched_random_window"}
                    continue
                trace = jax.tree.map(lambda x: x[i], traces)
                onset = (experiment.horizon_steps if i == 0 else
                         control_audit["random_delay"]["onset"] if i == 3 else experiment.event_step)
                # Receiver treatment cannot affect a preceding transition.
                # Include actions, actual cache/receipts, physical traces and RNG.
                pre_intervention = tree_comparison(jax.tree.map(lambda x: x[:onset], reference),
                                                    jax.tree.map(lambda x: x[:onset], trace))
                if not pre_intervention["passed"]:
                    write_json(directory/f"{name}_pre_intervention_failure.json", pre_intervention)
                    raise AssertionError("A branch diverged before its communication intervention")
                metrics = summarize_branch(trace, experiment)
                scores = metrics.pop("score_trace")
                np.savez_compressed(directory/f"{name}_trace.npz", **trace, score_trace=np.asarray(scores))
                branch_final = jax.tree.map(lambda x: x[i], final)
                with (directory/f"{name}_final.pkl").open("wb") as stream:
                    pickle.dump(branch_final, stream, protocol=5)
                branches[name] = {"status": "completed", "metrics": metrics, "fork_snapshot_sha256": snapshot_hash,
                    "pre_intervention_alignment": pre_intervention,
                    "trace_sha256": file_sha256(directory/f"{name}_trace.npz"),
                    "final_state_sha256": file_sha256(directory/f"{name}_final.pkl")}
            exposure_match = compare_exposure(branches["delay_up"]["metrics"]["exposure"]["up"],
                branches["delay_down"]["metrics"]["exposure"]["down"],
                first_window_steps=experiment.window_steps, second_window_steps=experiment.window_steps)
            direction = case["control_direction"]
            random_match = None
            if available[3]:
                random_match = compare_exposure(branches[f"delay_{direction}"]["metrics"]["exposure"][direction],
                    branches["random_delay"]["metrics"]["exposure"][direction],
                    first_window_steps=experiment.window_steps, second_window_steps=experiment.window_steps)
            summary = {**prefix_status, "status": "completed", "branches": branches,
                "snapshot_sha256": snapshot_hash, "intact_policy_alignment": alignment, "intact_replay": replay,
                "reference_slot_alignment": reference_slot_checks,
                "reference_slot_arrays_exactly_equal": reference_slot_exact,
                "directional_exposure_match": exposure_match, "random_exposure_match": random_match,
                "formal_result": False, "clean_direction_causal_claim_allowed": False,
                "claim_reason": "Development technical experiment; exposure matching alone is insufficient for a causal conclusion",
                "elapsed_s": time.monotonic()-case_started}
            write_json(directory/"summary.json", summary)
            summaries.append(summary)
            print(json.dumps({"case_id": case["case_id"], "status": summary["status"], "elapsed_s": summary["elapsed_s"],
                              "directional_exposure": exposure_match["status"]}), flush=True)
        result.update(status="completed_development_only", expected_cases=len(suite["cases"]),
            completed_cases=sum(s["status"] == "completed" for s in summaries),
            unavailable_forks=sum(s["status"] == "unavailable" for s in summaries),
            available_branch_counts={b: sum(s["branches"][b]["status"] == "completed" for s in summaries) for b in BRANCHES},
            case_summaries=summaries)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result["checkpoint_unchanged"] = file_sha256(args.checkpoint) == args.checkpoint_sha256
        result["elapsed_s"] = time.monotonic()-started
        write_json(args.output/"result.json", result)
        if not result["checkpoint_unchanged"]:
            raise AssertionError("Frozen producer checkpoint changed during E3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/e3_development_smoke.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
