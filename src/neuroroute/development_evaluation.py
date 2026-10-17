"""Frozen mixed and Central PPO on development-only fixed exogenous cases.

The CLI imports the simulator only after requiring a GPU Slurm allocation.
This does not optimize, update normalization, evaluate formal seeds, or support
foreign/published-policy checkpoint formats. Formats1/2 have explicit separate
validation and array-typing paths. Load trusted local pickle files only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import time

from .evaluation_cases import build_suite, canonical, digest, keyed_seed, key_words, write_immutable
from .runtime import file_sha256, require_compute, write_json


def mixed_runtime_candidate(content):
    """Select an explicit checkpoint identity without importing a GPU backend."""
    version=content.get("format_version")
    runtime=content.get("numerical_runtime")
    if version==1:
        if runtime is not None:
            raise ValueError("Legacy version1 cannot carry a version2 runtime identity")
        return "legacy_unbound"
    if version!=2 or not isinstance(runtime,dict):
        raise ValueError("Unsupported checkpoint format/runtime; no implicit migration")
    candidate=runtime.get("runtime_candidate")
    if candidate is None:
        return "canonical_scatter_workaround"
    if candidate=="jax082_fast_scatter":
        return candidate
    raise ValueError("Unknown version2 runtime candidate; no fallback")


def evaluation_numerical_runtime(candidate):
    if candidate=="jax082_fast_scatter":
        from .fast_reproducible_training import fast_numerical_runtime
        return fast_numerical_runtime()
    if candidate in ("canonical_scatter_workaround","legacy_unbound"):
        from .reproducible_training import numerical_runtime
        return numerical_runtime()
    raise ValueError("Unknown evaluation runtime candidate")


def runtime_metadata(env):
    """Allowlisted execution settings only; never dump the process environment."""
    import importlib.metadata
    import jax
    names = ["XLA_FLAGS","JAX_ENABLE_X64","JAX_DEFAULT_MATMUL_PRECISION",
             "JAX_ENABLE_COMPILATION_CACHE","JAX_COMPILATION_CACHE_DIR","JAX_THREEFRY_PARTITIONABLE",
             "CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES","NVIDIA_TF32_OVERRIDE",
             "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","SLURM_CPUS_PER_TASK"]
    versions = {}
    for name in ["jax","jaxlib","numpy","flax","mujoco","mujoco-mjx","brax"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        result = subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10,check=False)
        driver = {"returncode":result.returncode,"stdout":result.stdout.strip(),"stderr":result.stderr.strip()}
    except (FileNotFoundError,subprocess.TimeoutExpired) as exc:
        driver = {"error":type(exc).__name__}
    asset_hashes = {name:hashlib.sha256(data).hexdigest() for name,data in sorted(env._model_assets.items())}
    return {"python":platform.python_version(),"hostname":platform.node(),"packages":versions,
        "devices":[{"id":d.id,"platform":d.platform,"device_kind":d.device_kind} for d in jax.devices()],
        "settings":{name:os.environ.get(name) for name in names},
        "jax_effective_enable_x64":bool(jax.config.jax_enable_x64),
        "jax_effective_matmul_precision":str(jax.config.jax_default_matmul_precision),
        "nvidia_smi":driver,"model_xml":str(env._xml_path),"model_xml_sha256":file_sha256(env._xml_path),
        "model_asset_sha256":asset_hashes,"model_asset_manifest_sha256":digest(asset_hashes)}


def compare_replay(reference, replay, *, atol=1e-6, rtol=1e-6):
    """Full-trace diagnostics, not a declaration that chaotic paths must match."""
    import numpy as np
    if reference.keys()!=replay.keys():
        raise ValueError("Replay trace fields differ")
    fields={}
    for name in reference:
        a,b=np.asarray(reference[name]),np.asarray(replay[name])
        if a.shape!=b.shape:
            raise ValueError(f"Replay shape changed for {name}")
        same=np.array_equal(a,b,equal_nan=True)
        difference=np.abs(a.astype(np.float64)-b.astype(np.float64))
        finite=difference[np.isfinite(difference)]
        fields[name]={"array_equal_nan_equal":bool(same),
            "within_tolerance":bool(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True)),
            "max_abs_difference_finite":float(finite.max()) if finite.size else None,
            "nonfinite_pattern_equal":bool(np.array_equal(np.isfinite(a),np.isfinite(b)))}
    return {"atol":atol,"rtol":rtol,"all_arrays_equal_nan_equal":all(v["array_equal_nan_equal"] for v in fields.values()),
        "all_within_tolerance":all(v["within_tolerance"] for v in fields.values()),"fields":fields,
        "interpretation":"same-checkpoint same-case same-batch same-executable replay; includes inactive padding diagnostics"}


def select_cases(suite, case_ids=None, kinds=("healthy",), limit=None):
    if suite.get("split") != "development":
        raise ValueError("This runner only permits the development case namespace")
    if case_ids:
        wanted = set(case_ids)
        result = [c for c in suite["cases"] if c["case_id"] in wanted]
        if {c["case_id"] for c in result} != wanted:
            raise ValueError("Unknown case ID; never silently replace unavailable cases")
    else:
        result = [c for c in suite["cases"] if c["kind"] in kinds]
    if limit is not None:
        if limit <= 0:
            raise ValueError("Case limit must be positive")
        result = result[:limit]
    if not result:
        raise ValueError("Empty case selection")
    for case in result:
        original = {k:v for k,v in case.items() if k != "case_sha256"}
        if case.get("case_sha256") != digest(original):
            raise ValueError("Evaluation case checksum mismatch")
    return result


def case_arrays(cases, method, training_seed):
    import numpy as np
    n = cases[0]["horizon_steps"]
    if any(c["horizon_steps"] != n or c["dt_s"] != cases[0]["dt_s"] for c in cases):
        raise ValueError("A batch requires a common fixed horizon and dt")
    commands, environment_keys, policy_keys = [], [], []
    policy_namespace = f"neuroroute.development.policy.v1:{method}:{training_seed}"
    for c in cases:
        dense = np.zeros((n+1,3),np.float32)
        for entry in c["command_script"]:
            dense[entry["step"]:] = entry["command"]
        commands.append(dense)
        environment_keys.append([key_words(keyed_seed(c["namespace"],c["script_id"],"observation_noise",step)) for step in range(n)])
        policy_keys.append([key_words(keyed_seed(policy_namespace,c["script_id"],"communication",step)) for step in range(n)])
    return {"commands": np.stack(commands,axis=1),
        "environment_keys": np.asarray(environment_keys,np.uint32).transpose(1,0,2),
        "policy_keys": np.asarray(policy_keys,np.uint32).transpose(1,0,2),
        "reset_keys": np.asarray([key_words(c["reset_seed_u64"]) for c in cases],np.uint32),
        "fault_kind": np.asarray([{"healthy":0,"rough":0,"actuator":1,"sensor_freeze":2}[c["kind"]] for c in cases],np.int32),
        "leg": np.asarray([c["fault"]["leg_index"] if c["fault"] else 0 for c in cases],np.int32),
        "onset": np.asarray([c["fault"]["onset_step"] if c["fault"] else n+1 for c in cases],np.int32),
        "duration": np.asarray([c["fault"]["duration_steps"] if c["fault"] else 0 for c in cases],np.int32),
        "loss": np.asarray([c["fault"].get("loss_fraction") or 0 if c["fault"] else 0 for c in cases],np.float32),
        "policy_namespace": policy_namespace}


def validate_rough_domain(protocol, path, expected_sha256, env, runtime_candidate="canonical_scatter_workaround"):
    """Accept only the separately audited, pinned static official domain."""
    spec=protocol.get("rough_terrain",{})
    if spec.get("kind")!="official_static_heightfield" or spec.get("terrain_seed_role")!="reserved_unused" or spec.get("distinct_terrain_assets")!=1:
        raise ValueError("Use the explicit version2 static-rough protocol; no implicit terrain generator")
    if not expected_sha256 or file_sha256(path)!=expected_sha256:
        raise ValueError("Rough compatibility artifact hash mismatch")
    audit=json.loads(Path(path).read_text())
    if audit.get("status")!="compatible_fixed_official_rough" or not audit.get("checks") or not all(audit["checks"].values()):
        raise ValueError("Official rough compatibility checks have not passed")
    if audit.get("numerical_runtime")!=evaluation_numerical_runtime(runtime_candidate):
        raise ValueError("Rough audit and evaluation numerical runtime differ")
    recorded=audit["domains"]["rough_terrain"]
    current_asset=hashlib.sha256(env._model_assets[Path(spec["asset_file"]).name]).hexdigest()
    if (file_sha256(env._xml_path)!=spec["scene_sha256"] or current_asset!=spec["asset_sha256"]
            or recorded["runtime_and_assets"]["model_xml_sha256"]!=spec["scene_sha256"]
            or recorded["runtime_and_assets"]["model_asset_sha256"]["hfield.png"]!=spec["asset_sha256"]
            or recorded["config"]!=env._config.to_dict()):
        raise ValueError("Current rough scene, heightfield, or configuration differs from pinned audit")
    return {"status":"verified_static_official_domain","compatibility_path":str(Path(path).resolve()),
        "compatibility_sha256":expected_sha256,"asset_sha256":spec["asset_sha256"],
        "scene_sha256":spec["scene_sha256"],"terrain_seed_role":"reserved_unused","distinct_terrain_assets":1,
        "domain_changes":spec["domain_changes"]}


def make_rollout(env, actor, config, horizon, save_state_trace=False, canonicalize=False):
    import jax
    import jax.numpy as jnp
    from mujoco_playground._src.locomotion.go1.joystick import Joystick
    from .environment import _ModelView, scale_actuators, split_observations, assemble_actions
    from .communication import empty_state, full_gates, periodic_gates
    from .training import initialize_history, append_history, select_environments
    central=config.method=="B_CENTRAL"
    if central:
        from .centralized_training import new_history,advance_history
    if canonicalize:
        from .reproducible_training import canonical_arrays
    else:
        canonical_arrays = lambda tree: tree

    def command_override(state, command):
        info = dict(state.info)
        info["command"] = command
        # Upstream command generation may still consume RNG but cannot change
        # the current script command: this counter is replaced each step.
        info["steps_until_next_cmd"] = jnp.asarray(horizon+1,jnp.int32)
        obs = dict(state.obs)
        obs["state"] = obs["state"].at[45:48].set(command)
        obs["privileged_state"] = obs["privileged_state"].at[45:48].set(command)
        return state.replace(info=info,obs=obs)

    def reset_one(key, command):
        state = env.reset(key)
        state.info["nr_eval_frozen"] = jnp.zeros(6)
        return command_override(state,command)

    def physics_step(state, action, rng, command, next_command, step, kind, leg, onset, duration, loss):
        state = state.replace(info=dict(state.info),metrics=dict(state.metrics))
        state = command_override(state,command)
        state.info["rng"] = rng
        actuator_active = (kind == 1) & (step >= onset)
        scale = jnp.ones(12).at[jnp.asarray(env.leg_map.actuator)[leg]].set(jnp.where(actuator_active,1-loss,1.))
        model = scale_actuators(env.mjx_model,scale)
        nxt = Joystick.step(_ModelView(env,model),state,action)
        sensor_indices = jnp.concatenate((9+jnp.asarray(env.leg_map.qpos_local)[leg],21+jnp.asarray(env.leg_map.qvel_local)[leg]))
        reported = nxt.obs["state"][sensor_indices]
        frozen = jnp.where(step+1 == onset,reported,state.info["nr_eval_frozen"])
        freeze_next = (kind == 2) & (step+1 >= onset) & (step+1 < onset+duration)
        patched = nxt.obs["state"].at[sensor_indices].set(jnp.where(freeze_next,frozen,reported))
        nxt = nxt.replace(obs={**nxt.obs,"state":patched})
        nxt.info["nr_eval_frozen"] = frozen
        nxt = command_override(nxt,next_command)
        fault_active = actuator_active | ((kind == 2) & (step >= onset) & (step < onset+duration))
        freeze_error = jnp.where(freeze_next,jnp.max(jnp.abs(nxt.obs["state"][sensor_indices]-frozen)),0.)
        return nxt, (fault_active,freeze_error)

    def rollout(params, arrays):
        batch = arrays["reset_keys"].shape[0]
        state = canonical_arrays(jax.vmap(reset_one)(arrays["reset_keys"],arrays["commands"][0]))
        initial = state
        history = new_history(state.obs) if central else initialize_history(*split_observations(state.obs,env.leg_map),config.history_length)
        cache = () if central else empty_state((batch,))
        live = jnp.ones(batch,jnp.bool_)

        def step_fn(carry, step):
            state,history,cache,live = carry
            if config.method == "B_FULL":
                forced = full_gates((batch,))
            elif config.method == "B_PERIODIC":
                forced = periodic_gates(jnp.full((batch,),step),config.budget_fraction)
            else:
                forced = jnp.zeros((batch,8),jnp.bool_)
            def policy_one(g,l,c,k,f):
                return actor.apply({"params":params},g,l,c,k,env.dt,
                    forced_gates=f if config.method in {"B_FULL","B_PERIODIC"} else None,
                    deterministic_motor=True)
            if central:
                output=jax.vmap(lambda h,k:actor.apply({"params":params},h,k,deterministic_motor=True))(
                    history,arrays["policy_keys"][step])
            else:
                global_obs,local_obs = history.actor_inputs()
                output = jax.vmap(policy_one)(global_obs,local_obs,cache,arrays["policy_keys"][step],forced)
            # Batch-index-independent policy keys prevent padding/batch-size from
            # changing any case's sampled communication actions.
            nxt,(fault_active,freeze_error) = jax.vmap(physics_step,in_axes=(0,0,0,0,0,None,0,0,0,0,0))(
                state,assemble_actions(output.motor_action,env.leg_map),arrays["environment_keys"][step],
                arrays["commands"][step],arrays["commands"][step+1],step,
                arrays["fault_kind"],arrays["leg"],arrays["onset"],arrays["duration"],arrays["loss"])
            nxt = canonical_arrays(nxt)
            velocity = jax.vmap(env.get_local_linvel)(nxt.data)[:,:2]
            yaw = jax.vmap(env.get_gyro)(nxt.data)[:,2]
            finite = (jnp.all(jnp.isfinite(nxt.data.qpos),axis=-1) & jnp.all(jnp.isfinite(nxt.data.qvel),axis=-1)
                      & jnp.all(jnp.isfinite(nxt.obs["state"]),axis=-1)
                      & jnp.all(jnp.isfinite(velocity),axis=-1) & jnp.isfinite(yaw)
                      & jnp.all(jnp.isfinite(output.motor_action),axis=(-1,-2)))
            terminated = nxt.done.astype(jnp.bool_) | ~finite
            alive_next = live & ~terminated
            next_history = advance_history(history,nxt.obs) if central else append_history(history,*split_observations(nxt.obs,env.leg_map))
            record = {"velocity_xy":velocity,"yaw_rate":yaw,"command":arrays["commands"][step],
                "active_before_step":live,"terminated_this_step":live & terminated,"nonfinite_this_step":live & ~finite,
                "fault_active":fault_active & live,"freeze_error":jnp.where(live,freeze_error,0),
                "actuator_force":nxt.data.actuator_force,"motor_action":output.motor_action,
                "actor_command_error":jnp.max(jnp.abs((history[:,-1,-3:] if central else history.global_obs[:,-1,-3:])-arrays["commands"][step]),axis=-1),
                "raw_reward":jnp.where(live,nxt.reward,0)}
            if not central:
                gates=jnp.stack((output.action.up_gate,output.action.down_gate),axis=1)
                payload=jnp.stack((output.action.up_payload,output.action.down_payload),axis=1)
                record.update(sent_gates=gates & live[:,None,None],
                    sent_payload=jnp.where(live[:,None,None,None],payload,0),
                    cache_age_s=jnp.stack((output.cache.up.age,output.cache.down.age),axis=1))
            if save_state_trace:
                record.update(qpos=nxt.data.qpos,qvel=nxt.data.qvel)
            # Inactive slots are padding, not reset evaluation episodes. No new
            # live flag is created and all subsequent transmissions are masked.
            safe = select_environments(alive_next,nxt,initial)
            safe_history = select_environments(alive_next,next_history,history)
            return (safe,safe_history,() if central else output.cache,alive_next),record
        _,trace = jax.lax.scan(step_fn,(state,history,cache,live),jnp.arange(horizon))
        return trace
    return jax.jit(rollout)


def summarize_case(case, trace, protocol, config):
    import numpy as np
    from .metrics import tracking_metrics,recovery_metrics,bandwidth_metrics
    failed = np.flatnonzero(trace["terminated_this_step"])
    failure = int(failed[0]) if len(failed) else None
    score = tracking_metrics(trace["velocity_xy"],trace["yaw_rate"],trace["command"],
        horizon_steps=case["horizon_steps"],dt_s=case["dt_s"],failure_step=failure,
        sigma_v=protocol["score"]["sigma_v_m_s"],sigma_w=protocol["score"]["sigma_w_rad_s"])
    score_trace = score.pop("score_trace")
    active_steps = int(np.sum(trace["active_before_step"]))
    central=getattr(config,"method",None)=="B_CENTRAL"
    if central:
        if any(k in trace for k in ("sent_gates","sent_payload","cache_age_s")):
            raise ValueError("Central trace must not manufacture packet or cache observations")
        wire={"status":"not_applicable","reason":"centralized_full_observation_reference_has_no_packet_interface",
              "active_bits_per_step":None,"total_bits":None,"budget_bits_per_step":None}
    else:
        gates = trace["sent_gates"][:active_steps]
        wire = bandwidth_metrics(gates,dt_s=case["dt_s"],delivered_gates=gates,
            received_flags=np.ones_like(gates),planned_horizon_steps=case["horizon_steps"],
            budget_bits_per_step=520*config.budget_fraction)
        ages = trace["cache_age_s"][:active_steps]
        wire["cache_age_s_quantiles"] = {str(q):float(np.quantile(ages,q)) for q in [.5,.9,.99]}
    recovery = None
    if case["fault"]:
        recovery = recovery_metrics(score_trace,event_step=case["fault"]["onset_step"],dt_s=case["dt_s"],
            failure_step=failure,score_threshold=protocol["recovery"]["score_threshold"],hold_s=protocol["recovery"]["hold_s"])
        applied = bool(np.any(trace["fault_active"]))
        # Frozen generic metrics use sample-index boundary <=. Physical onset
        # occurs before the corresponding transition; same-transition falls
        # are after application. Preserve both conventions explicitly.
        recovery["sample_index_pre_event_classification"] = recovery["failed_before_event"]
        recovery["fault_applied_before_failure"] = applied
        if recovery["failed_before_event"] and applied:
            recovery.update(failed_before_event=False,status="failed_after_event",censor_time_s=case["dt_s"])
    reward = np.asarray(trace["raw_reward"])
    freeze_error = np.asarray(trace["freeze_error"][:active_steps])
    return {"case_id":case["case_id"],"case_sha256":case["case_sha256"],"kind":case["kind"],
        "script_id":case["script_id"],"fault":case["fault"],"tracking":score,"communication":wire,
        "recovery":recovery,"nonfinite_failure":bool(np.any(trace["nonfinite_this_step"])),
        "termination_transition_end_s":(failure+1)*case["dt_s"] if failure is not None else None,
        "raw_return":float(np.sum(reward)) if np.isfinite(reward).all() else None,
        "max_actor_command_error":float(np.max(trace["actor_command_error"][:active_steps])),
        "max_sensor_freeze_error":float(np.max(freeze_error)) if np.isfinite(freeze_error).all() else None,
        "policy_execution":"conditional_mean_motor_no_packet_interface" if central else "conditional_mean_motor_sampled_gate_and_payload",
        "phase":"development","formal_result":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--checkpoint-sha256",required=True)
    parser.add_argument("--protocol",type=Path,default=Path("configs/evaluation_protocol.json"))
    parser.add_argument("--suite",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case-id",action="append")
    parser.add_argument("--kinds",nargs="+",choices=["healthy","actuator","sensor_freeze","rough"],default=["healthy"])
    parser.add_argument("--limit",type=int)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--save-state-trace",action="store_true")
    parser.add_argument("--replays",type=int,default=1)
    parser.add_argument("--replay-atol",type=float,default=1e-6)
    parser.add_argument("--replay-rtol",type=float,default=1e-6)
    parser.add_argument("--rough-compatibility",type=Path)
    parser.add_argument("--rough-compatibility-sha256")
    args=parser.parse_args()
    require_compute(gpu=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .environment import FaultGo1,observation_manifest
    from .training import TrainConfig,implementation_fingerprint
    from .policy import HybridPolicy
    if args.batch_size<1 or args.replays<1 or min(args.replay_atol,args.replay_rtol)<0:
        raise ValueError("Batch size/replay count must be positive; tolerances nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh evaluation output; results are immutable")
    protocol=json.loads(args.protocol.read_text())
    expected_suite=build_suite(protocol,"development")
    suite=json.loads(args.suite.read_text()) if args.suite else expected_suite
    if suite!=expected_suite:
        raise ValueError("Supplied development suite differs from the frozen protocol")
    selected=select_cases(suite,args.case_id,args.kinds,args.limit)
    actual_hash=file_sha256(args.checkpoint)
    if len(args.checkpoint_sha256)!=64 or actual_hash!=args.checkpoint_sha256:
        raise ValueError("Checkpoint missing or changed from requested frozen SHA256")
    with args.checkpoint.open("rb") as stream:
        content=pickle.load(stream)
    if file_sha256(args.checkpoint)!=actual_hash:
        raise ValueError("Checkpoint changed while loading; evaluate an immutable snapshot")
    central=content.get("format")=="centralized_ppo_v1"
    if central:
        from .centralized_training import CentralConfig,CentralActor,CentralPPOTrainer
        config=CentralConfig(**content["config"])
    else:
        config=TrainConfig(**content["config"])
    config.validate()
    checkpoint_format=content["format"] if central else content.get("format_version")
    runtime_candidate=mixed_runtime_candidate({"format_version":2,"numerical_runtime":content.get("numerical_runtime")} if central else content)
    canonical_checkpoint=central or checkpoint_format==2
    if content.get("normalizer")!={"kind":"fixed_identity","updates":0}:
        raise ValueError("Unaudited checkpoint normalizer")
    env=FaultGo1(impl="jax")
    if content.get("env_class" if central else "environment_class")!=type(env).__module__+"."+type(env).__qualname__ or content["env_dt"]!=env.dt or env.dt!=protocol["control_dt_s"]:
        raise ValueError("Checkpoint/environment/protocol schema or dt mismatch")
    if central:
        if content.get("communication_cost")!="not_applicable" or content.get("scheduler")!={"kind":"constant","learning_rate":config.learning_rate}:
            raise ValueError("Unaudited Central scheduler or communication identity")
        if runtime_candidate=="jax082_fast_scatter":
            from .fast_centralized_training import FastCentralPPOTrainer
            validator=FastCentralPPOTrainer(env,config)
        else:
            validator=CentralPPOTrainer(env,config)
        restored=validator.load_checkpoint(args.checkpoint)
        if file_sha256(args.checkpoint)!=actual_hash:
            raise ValueError("Central checkpoint changed during producer validation")
        params=restored.params["actor"]
        del restored,validator
        checkpoint_runtime=content["numerical_runtime"]
    elif checkpoint_format==2:
        from .reproducible_training import ReproduciblePPOTrainer,canonical_arrays
        # Constructor validates the exact numerical runtime and derives the
        # authoritative v2 fingerprint without initializing/training any state.
        if runtime_candidate=="jax082_fast_scatter":
            from .fast_reproducible_training import FastReproduciblePPOTrainer
            validator=FastReproduciblePPOTrainer(env,config)
        else:
            validator=ReproduciblePPOTrainer(env,config)
        # Use the checkpoint producer's complete validator, including its
        # scheduler/config/environment contract, without initializing training.
        restored=validator.load_checkpoint(args.checkpoint)
        if file_sha256(args.checkpoint)!=actual_hash:
            raise ValueError("Checkpoint changed during strict producer validation")
        params=restored.params["actor"]
        del restored,validator
        checkpoint_runtime=content["numerical_runtime"]
    else:
        if content.get("implementation_sha256")!=implementation_fingerprint():
            raise ValueError("Legacy version1 core fingerprint differs")
        params=jax.tree.map(jnp.asarray,content["state"].params["actor"])
        checkpoint_runtime=None
    training_steps=int(content["state"].environment_steps)
    checkpoint_implementation=content["implementation_sha256"]
    del content
    rough_env=None
    rough_domain={"status":"blocked_until_explicit_static_domain_protocol_and_compatibility_artifact"}
    if any(c["kind"]=="rough" for c in selected) and args.rough_compatibility:
        rough_env=FaultGo1(task="rough_terrain",impl="jax")
        rough_domain=validate_rough_domain(protocol,args.rough_compatibility,args.rough_compatibility_sha256,rough_env,runtime_candidate)
    args.output.mkdir(parents=True,exist_ok=True)
    write_immutable(args.output/"development_suite.json",suite)
    write_json(args.output/"observation_schema.json",observation_manifest(env.leg_map))
    provenance={"phase":"development","formal_result":False,"method":config.method,"training_seed":config.seed,
        "checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":actual_hash,"training_environment_steps":training_steps,
        "training_core_sha256":implementation_fingerprint(),"protocol_file_sha256":file_sha256(args.protocol),
        "checkpoint_format_version":checkpoint_format,"checkpoint_numerical_runtime":checkpoint_runtime,
        "runtime_candidate":runtime_candidate,
        "checkpoint_family":"central" if central else "mixed",
        "checkpoint_implementation_sha256":checkpoint_implementation,
        "canonical_strong_arrays":canonical_checkpoint,
        "checkpoint_runtime_binding":"strict_producer_runtime" if canonical_checkpoint else "legacy_version1_runtime_unbound",
        "protocol_canonical_sha256":digest(protocol),"suite_cases_sha256":suite["cases_sha256"],
        "evaluator_sha256":file_sha256(__file__),"metric_and_generator_sha256":protocol["source_sha256"],
        "selected_case_ids":[c["case_id"] for c in selected],"selected_cases_sha256":digest(selected),
        "job_id":os.environ["SLURM_JOB_ID"],"batch_size":args.batch_size,"save_state_trace":args.save_state_trace,
        "evaluator_version":"development-evaluator-v6-central-and-mixed",
        "replays":args.replays,"replay_atol":args.replay_atol,"replay_rtol":args.replay_rtol,
        "runtime":runtime_metadata(env),
        "environment_noise":"absolute_step_keyed_exogenous_rng","policy_rng":"method_and_training_seed_namespace_percase_script_absolute_step",
        "environment_config":env._config.to_dict(),"rough_domain":rough_domain,
        "domain_environment_configs":{"flat_terrain":env._config.to_dict(),
            "rough_terrain":rough_env._config.to_dict() if rough_env is not None else None},
        "command_schedule_override":True,"normalization_updates":0}
    write_json(args.output/"provenance.json",provenance)
    flat_cases=[c for c in selected if c["kind"]!="rough"]
    rough_cases=[c for c in selected if c["kind"]=="rough"] if rough_env is not None else []
    supported=flat_cases+rough_cases
    records=[{"case_id":c["case_id"],"kind":"rough","status":rough_domain["status"],"phase":"development","formal_result":False} for c in selected if c["kind"]=="rough" and rough_env is None]
    actor=CentralActor(config.actor_width) if central else HybridPolicy(config.local_width,config.global_width)
    batches=[]
    total_offset=0
    for domain_env,group in [(env,flat_cases),(rough_env,rough_cases)]:
        if group:
            domain_rollout=make_rollout(domain_env,actor,config,selected[0]["horizon_steps"],args.save_state_trace,canonical_checkpoint)
            for offset in range(0,len(group),args.batch_size):
                cases=group[offset:offset+args.batch_size]
                batches.append((total_offset,cases,domain_rollout))
                total_offset+=len(cases)
    started=time.monotonic()
    replay_diagnostics=[]
    with (args.output/"cases.jsonl").open("x") as output:
        for record in records:
            output.write(json.dumps(record,allow_nan=False)+"\n")
        for offset,cases,rollout in batches:
            # Padding reuses a known case but is never emitted as another trial.
            padded=cases+[cases[-1]]*(args.batch_size-len(cases))
            arrays=case_arrays(padded,config.method,config.seed)
            policy_namespace=arrays.pop("policy_namespace")
            arrays=jax.tree.map(jnp.asarray,arrays)
            batch_started=time.monotonic()
            trace=jax.device_get(rollout(params,arrays))
            trace={k:v[:,:len(cases)] for k,v in trace.items()}
            path=args.output/f"trace_{offset:05d}.npz"
            np.savez_compressed(path,**trace)
            trace_hash=file_sha256(path)
            for i,case in enumerate(cases):
                record=summarize_case(case,{k:v[:,i] for k,v in trace.items()},protocol,config)
                record.update(trace_file=path.name,trace_batch_index=i,trace_sha256=trace_hash,policy_rng_namespace=policy_namespace)
                if case["kind"]=="rough":
                    record["rough_domain"]=rough_domain
                if record["max_actor_command_error"]>1e-6 or (record["max_sensor_freeze_error"] is not None and record["max_sensor_freeze_error"]>1e-6):
                    raise AssertionError("Command/frozen observation contract violated")
                records.append(record)
                output.write(json.dumps(record,allow_nan=False)+"\n")
            for repetition in range(1,args.replays):
                repeated=jax.device_get(rollout(params,arrays))
                repeated={k:v[:,:len(cases)] for k,v in repeated.items()}
                repeated_path=args.output/f"trace_{offset:05d}_replay_{repetition:03d}.npz"
                np.savez_compressed(repeated_path,**repeated)
                comparison=compare_replay(trace,repeated,atol=args.replay_atol,rtol=args.replay_rtol)
                repeated_records=[summarize_case(case,{k:v[:,i] for k,v in repeated.items()},protocol,config)
                                  for i,case in enumerate(cases)]
                for r in repeated_records:
                    r.update(repetition=repetition,diagnostic_only=True,independent_trial=False)
                comparison.update(batch_offset=offset,repetition=repetition,trace_file=repeated_path.name,
                    trace_sha256=file_sha256(repeated_path),case_results=repeated_records)
                replay_diagnostics.append(comparison)
            output.flush()
            write_json(args.output/"progress.json",{"completed":offset+len(cases),"requested_supported":len(supported),"last_batch_s":time.monotonic()-batch_started})
            print(json.dumps({"completed":offset+len(cases),"batch_seconds":time.monotonic()-batch_started}),flush=True)
    summaries={}
    for kind in ["healthy","actuator","sensor_freeze","rough"]:
        group=[r for r in records if r["kind"]==kind and "tracking" in r]
        if group:
            summaries[kind]={"n":len(group),"mean_score":float(np.mean([r["tracking"]["score"] for r in group])),
                "completed_fraction":float(np.mean([r["tracking"]["completed"] for r in group])),
                "mean_active_bits_per_step":None if central else float(np.mean([r["communication"]["active_bits_per_step"] for r in group])),
                "communication_status":"not_applicable" if central else "measured_packet_interface",
                "nonfinite_failures":sum(r["nonfinite_failure"] for r in group)}
    summary={**provenance,"status":"completed_development_evaluation","elapsed_s":time.monotonic()-started,
        "completed_cases":len(supported),"blocked_cases":len(selected)-len(supported),"by_kind":summaries,
        "cases_jsonl_sha256":file_sha256(args.output/"cases.jsonl"),
        "interpretation":"development diagnostics only; no formal comparison or inferential interval"}
    write_json(args.output/"replay_diagnostics.json",{"requested_replays":args.replays,
        "comparisons":replay_diagnostics,"independent_additional_trials":0})
    summary["replay_diagnostics_sha256"]=file_sha256(args.output/"replay_diagnostics.json")
    write_json(args.output/"summary.json",summary)
    print(json.dumps({"status":summary["status"],"by_kind":summaries}),flush=True)


if __name__=="__main__":
    main()
