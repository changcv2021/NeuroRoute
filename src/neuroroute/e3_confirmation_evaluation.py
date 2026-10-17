"""Version3 E3 evaluation with independent formal and technical case namespaces.

Uses the frozen numerical fork engine. Every formal run evaluates all 80 fixed
forks; unavailable forks and unmatched actual exposure remain explicit.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time
from .runtime import require_compute, write_json, file_sha256
from .e3_evaluation import make_rollout, empty_controls, tree_comparison, summarize_branch
from .e3_confirmation_cases import (BRANCHES, METHOD_BUDGETS, ConfirmationE3Design,
    load_protocol, build_cases, dense_scripts, branch_controls, select_shard)


def run(args):
    require_compute(gpu=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .environment import FaultGo1
    from .confirmation_protocol import ConfirmationConfig
    from .confirmation_training import make_confirmation_trainer
    from .confirmation_evaluation import phase_gate
    from .policy import HybridPolicy
    from .e3_policy import InterventionPolicy
    from .communication import full_gates
    from .e3_transport import compare_exposure
    from .development_evaluation import runtime_metadata

    started = time.monotonic()
    protocol, _ = load_protocol(args.protocol, args.protocol_sha256)
    suite = select_shard(build_cases(args.protocol_sha256, phase=args.phase, smoke=args.smoke),
                         args.shard_index, smoke=args.smoke)
    experiment = ConfirmationE3Design(**suite["design"])
    formal = args.phase == "confirmation"
    if args.output.exists():
        raise FileExistsError("Use a new E3 output directory; prior branches are immutable")
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint differs from the explicitly frozen producer hash")
    args.output.mkdir(parents=True)
    write_json(args.output/"cases.json", suite)
    write_json(args.output/"configuration.json", asdict(experiment))
    result = {"status": "failed", "phase": args.phase, "formal_result": formal,
              "performance_claim": False, "checkpoint_sha256": args.checkpoint_sha256,
              "e3_protocol_sha256": args.protocol_sha256, "suite_cases_sha256": suite["suite_cases_sha256"],
              "shard_index": suite["shard_index"], "shard_count": suite["shard_count"],
              "suite_case_count": suite["suite_case_count"]}
    try:
        # Trusted locally generated checkpoint only. Never deserialize uploads.
        with args.checkpoint.open("rb") as stream:
            header = pickle.load(stream)
        phase_gate(header, args.phase)
        config = ConfirmationConfig(**header["config"])
        config.validate()
        resolved, selected = config.validate_manifests()
        if config.method not in METHOD_BUDGETS:
            raise ValueError("E3 predeclared methods are FULL, ASYM and BIO")
        if formal and config.budget_fraction != METHOD_BUDGETS[config.method]:
            raise ValueError("Method does not have its predeclared E3 budget")
        if formal and (config.protocol_sha256 != args.training_protocol_sha256
                       or config.selection_sha256 != args.selection_sha256):
            raise ValueError("E3 checkpoint does not match the registered producer")
        env = FaultGo1(impl="jax")
        if env._config.pert_config.enable or not np.isclose(env.dt, experiment.dt_s, rtol=0, atol=1e-10):
            raise ValueError("E3 requires audited dt and disabled independent random perturbations")
        validator = make_confirmation_trainer(env, config)
        loaded = validator.load_checkpoint(args.checkpoint)
        if formal and (int(loaded.environment_steps) != protocol["common_training_steps"]
                       or int(loaded.environment_steps) != header["training_environment_steps_target"]):
            raise ValueError("E3 requires the complete final predeclared training step")
        params = loaded.params["actor"]
        actor = InterventionPolicy(config.local_width, config.global_width, control_dt=env.dt)
        original_actor = HybridPolicy(config.local_width, config.global_width)
        reset, prefix_rollout, fork_rollout, event = make_rollout(env, actor, config, experiment)
        provenance = {"checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": args.checkpoint_sha256,
            "checkpoint_format": 3, "phase": args.phase, "formal_result": formal,
            "e3_protocol_sha256": args.protocol_sha256, "training_protocol_sha256": config.protocol_sha256,
            "selection_sha256": config.selection_sha256, "numerical_runtime": validator.runtime,
            "checkpoint_update_index": int(loaded.update_index), "checkpoint_env_steps": int(loaded.environment_steps),
            "training_method": config.method, "training_seed": config.seed,
            "environment": runtime_metadata(env), "push": event.manifest(),
            "push_strength_status": "fixed_20N_100ms_2Ns_physical_units_no_performance_tuning",
            "normalizer": header["normalizer"], "source_hashes": {p.name: file_sha256(p) for p in
                [Path(__file__), Path(__file__).with_name("e3_cases.py"), Path(__file__).with_name("e3_policy.py"),
                 Path(__file__).with_name("e3_transport.py"), Path(__file__).with_name("e3_push.py"), args.protocol]},
            "job_id": os.environ["SLURM_JOB_ID"], "rng_namespace": suite["namespace"],
            "motor_policy": "conditional_mean_with_sampled_messages_and_gates",
            "fork_execution": "serial_lax_map_shared_scalar_compiled_body",
            "actor_inputs": "only legal 3-frame global/local observations plus received packet cache",
            "formal_model_selection_resolved": formal, "clean_content_causal_claim_allowed": False,
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
                "formal_result": formal, "clean_direction_causal_claim_allowed": False,
                "claim_reason": "Report paired branch effects and measured exposure; exposure matching alone is insufficient for a causal conclusion",
                "elapsed_s": time.monotonic()-case_started}
            write_json(directory/"summary.json", summary)
            summaries.append(summary)
            print(json.dumps({"case_id": case["case_id"], "status": summary["status"], "elapsed_s": summary["elapsed_s"],
                              "directional_exposure": exposure_match["status"]}), flush=True)
        result.update(status="completed_confirmation_e3_shard" if formal else "completed_validation_e3", expected_cases=len(suite["cases"]),
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['confirmation', 'validation'], required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--training-protocol-sha256')
    p.add_argument('--selection-sha256')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.phase == 'confirmation' and (args.smoke or not args.training_protocol_sha256 or not args.selection_sha256):
        raise ValueError('Formal E3 requires all predeclared cases and exact producer protocol/selection')
    run(args)


if __name__ == '__main__':
    main()
