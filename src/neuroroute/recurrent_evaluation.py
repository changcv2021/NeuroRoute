"""Recursive MATCH evaluation with the frozen common Go1 physics and metrics.

Actor recurrence advances on legal observations at every control step. The
CLI accepts only development checkpoints; a separate strict confirmation
adapter must validate its own producer before calling the shared runner.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time

from .runtime import require_compute, file_sha256, write_json
from .evaluation_cases import build_suite, digest, write_immutable
from .confirmation_evaluation import SHARED_SOURCE_SHA256, phase_case_arrays
from .development_evaluation import summarize_case, runtime_metadata, compare_replay, validate_rough_domain

CORE_SHA = '8f1f0112abf12d24f104078e7ae3e3e7f8a85eac23a8c7469a7cb7af4ca201d2'
FORMAT = 'neuroroute.match.recursive_fullprefix.full_batch_forward.v5'


def validate_sources():
    for name, sha in SHARED_SOURCE_SHA256.items():
        if file_sha256(Path(__file__).with_name(name)) != sha:
            raise ValueError('Shared scientific evaluator changed: '+name)


def development_suite(protocol, phase):
    if phase == 'development':
        suite = build_suite(protocol, 'development')
        selected = [c for c in suite['cases'] if c['kind'] == 'healthy']
        if len(selected) != 100:
            raise ValueError('HPO requires the complete common healthy100 design')
    elif phase == 'validation':
        technical = dict(protocol)
        technical['development_namespace'] = 'neuroroute.validation.recurrent.eval.v1'
        suite = build_suite(technical, 'development')
        for case in suite['cases']:
            case['case_id'] = case['case_id'].replace('development:', 'validation:', 1)
            case['case_sha256'] = digest({k:v for k,v in case.items() if k != 'case_sha256'})
        suite.update(cases_sha256=digest(suite['cases']), split='validation', technical_validation=True)
        selected = suite['cases']
    else:
        raise ValueError('Development loader cannot select formal cases')
    return suite, selected


def make_rollout(env, actor, config, horizon, save_state_trace=False, canonicalize=False):
    import jax
    import jax.numpy as jnp
    from mujoco_playground._src.locomotion.go1.joystick import Joystick
    from .environment import _ModelView, scale_actuators, split_observations, assemble_actions
    from .communication import empty_state, full_gates, periodic_gates
    from .training import select_environments
    from .recurrent_published_policy import empty_recurrent_match_state
    if config.method != "B_PUBLISHED":
        raise ValueError("Recursive actor interface only")
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
        history = empty_recurrent_match_state((batch,),config.history_length,config.width)
        cache = ()
        live = jnp.ones(batch,jnp.bool_)

        def step_fn(carry, step):
            state,history,cache,live = carry
            global_obs, local_obs = split_observations(state.obs,env.leg_map)
            generated = jax.vmap(lambda g,l,h,k: actor.apply({"params":params},g,l,h,k,env.dt,
                deterministic_motor=True))(global_obs,local_obs,history,arrays["policy_keys"][step])
            output, next_history = generated.policy, generated.state
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
            record = {"velocity_xy":velocity,"yaw_rate":yaw,"command":arrays["commands"][step],
                "active_before_step":live,"terminated_this_step":live & terminated,"nonfinite_this_step":live & ~finite,
                "fault_active":fault_active & live,"freeze_error":jnp.where(live,freeze_error,0),
                "actuator_force":nxt.data.actuator_force,"motor_action":output.motor_action,
                "actor_command_error":jnp.max(jnp.abs(global_obs[:,-3:]-arrays["commands"][step]),axis=-1),
                "raw_reward":jnp.where(live,nxt.reward,0)}
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
            return (safe,safe_history,(),alive_next),record
        _,trace = jax.lax.scan(step_fn,(state,history,cache,live),jnp.arange(horizon))
        return trace
    return jax.jit(rollout)


def evaluate(*, env, actor, config, params, protocol, suite, chosen, phase, output,
             evidence, batch_size=16, replays=1, save_state_trace=False, rough_path=None, rough_sha=None):
    """No optimization; common exogenous physics, sticky termination and metrics."""
    require_compute(gpu=True)
    validate_sources()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .environment import FaultGo1, observation_manifest
    if batch_size != 16 or replays < 1 or phase not in ('development','validation','confirmation'):
        raise ValueError('Registered B16 and an explicit evaluation phase are required')
    if float(env.dt) != protocol['control_dt_s']:
        raise ValueError('Physics and frozen protocol control dt differ')
    formal = phase == 'confirmation'
    rough_env, rough_domain = None, None
    if any(c['kind'] == 'rough' for c in chosen):
        if not rough_path or not rough_sha:
            raise ValueError('Rough cases require the pinned compatibility artifact')
        rough_env = FaultGo1(task='rough_terrain',impl='jax')
        rough_domain = validate_rough_domain(protocol,rough_path,rough_sha,rough_env,'jax082_fast_scatter')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_immutable(output/'evaluation_suite.json',suite)
    write_json(output/'training_config.json',asdict(config))
    write_json(output/'observation_schema.json',observation_manifest(env.leg_map))
    provenance = {**evidence,'phase':phase,'formal_result':formal,'method':config.method,
        'training_seed':config.seed,'evaluator_sha256':file_sha256(__file__),
        'shared_source_sha256':SHARED_SOURCE_SHA256,'protocol_canonical_sha256':digest(protocol),
        'suite_cases_sha256':suite['cases_sha256'],'selected_cases_sha256':digest(chosen),
        'selected_case_ids':[c['case_id'] for c in chosen],'batch_size':batch_size,'replays':replays,
        'normalization_updates':0,'command_schedule_override':True,'canonical_strong_arrays':True,
        'runtime_candidate':'jax082_fast_scatter','checkpoint_family':'recursive_published',
        'recurrent_state':'fresh_zero_at_reset_then_full_causal_recursive_update_every_control_step',
        'runtime':runtime_metadata(env),'rough_domain':rough_domain,
        'environment_config':env._config.to_dict(),'job_id':os.environ['SLURM_JOB_ID']}
    write_json(output/'provenance.json',provenance)
    started = time.monotonic()
    records, comparisons = [], []
    offset = 0
    with (output/'cases.jsonl').open('x') as stream:
        for domain, group in [(env,[c for c in chosen if c['kind']!='rough']),
                              (rough_env,[c for c in chosen if c['kind']=='rough'])]:
            if not group:
                continue
            rollout = make_rollout(domain,actor,config,group[0]['horizon_steps'],save_state_trace,True)
            for index in range(0,len(group),batch_size):
                cases = group[index:index+batch_size]
                padded = cases+[cases[-1]]*(batch_size-len(cases))
                arrays = phase_case_arrays(padded,config.method,config.seed,phase)
                namespace = arrays.pop('policy_namespace')
                arrays = jax.tree.map(jnp.asarray,arrays)
                trace = jax.device_get(rollout(params,arrays))
                trace = {k:v[:,:len(cases)] for k,v in trace.items()}
                path = output/f'trace_{offset:05d}.npz'
                np.savez_compressed(path,**trace)
                trace_sha = file_sha256(path)
                for i,case in enumerate(cases):
                    record = summarize_case(case,{k:v[:,i] for k,v in trace.items()},protocol,config)
                    record.update(phase=phase,formal_result=formal,trace_file=path.name,
                        trace_batch_index=i,trace_sha256=trace_sha,policy_rng_namespace=namespace)
                    if case['kind']=='rough':
                        record['rough_domain']=rough_domain
                    if record['max_actor_command_error']>1e-6 or (
                        record['max_sensor_freeze_error'] is not None and record['max_sensor_freeze_error']>1e-6):
                        raise AssertionError('Common command or sensor-freeze contract violated')
                    records.append(record)
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
                for repetition in range(1,replays):
                    repeated = jax.device_get(rollout(params,arrays))
                    repeated = {k:v[:,:len(cases)] for k,v in repeated.items()}
                    comparison = compare_replay(trace,repeated,atol=1e-6,rtol=1e-6)
                    comparison.update(batch_offset=offset,repetition=repetition)
                    comparisons.append(comparison)
                    if not comparison['all_arrays_equal_nan_equal']:
                        raise AssertionError('Repeated same-checkpoint recursive evaluation differs')
                offset += len(cases)
                stream.flush()
                write_json(output/'progress.json',{'completed':offset,'requested':len(chosen)})
                print(json.dumps({'completed':offset,'elapsed_seconds':time.monotonic()-started}),flush=True)
    groups = {}
    for kind in ('healthy','actuator','sensor_freeze','rough'):
        group = [r for r in records if r['kind']==kind]
        if group:
            groups[kind]={'n':len(group),'mean_score':float(np.mean([r['tracking']['score'] for r in group])),
                'completed_fraction':float(np.mean([r['tracking']['completed'] for r in group])),
                'mean_active_bits_per_step':float(np.mean([r['communication']['active_bits_per_step'] for r in group])),
                'nonfinite_failures':sum(r['nonfinite_failure'] for r in group)}
    write_json(output/'replay_diagnostics.json',{'comparisons':comparisons,'independent_additional_trials':0})
    summary = {**provenance,'status':f'completed_{phase}_evaluation','completed_cases':len(records),
        'blocked_cases':len(chosen)-len(records),'by_kind':groups,'elapsed_s':time.monotonic()-started,
        'cases_jsonl_sha256':file_sha256(output/'cases.jsonl'),
        'replay_diagnostics_sha256':file_sha256(output/'replay_diagnostics.json')}
    write_json(output/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'by_kind':groups}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--phase',choices=['development','validation'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--replays',type=int,default=1)
    p.add_argument('--rough-compatibility',type=Path);p.add_argument('--rough-compatibility-sha256')
    args=p.parse_args()
    if file_sha256(args.protocol)!=args.protocol_sha256:
        raise ValueError('Common evaluation protocol changed')
    validate_sources()
    protocol=json.loads(args.protocol.read_text())
    suite,chosen=development_suite(protocol,args.phase)
    if args.phase=='development':
        healthy=Path(__file__).resolve().parents[2]/'manifests/hpo_healthy_cases.json'
        if digest(chosen)!=digest(json.loads(healthy.read_text())):
            raise ValueError('HPO cases differ from the precommitted common100')
    require_compute(gpu=True)
    from .environment import FaultGo1
    from .recurrent_published_training_v5 import RecurrentMatchConfig,RecurrentMatchTrainer
    if file_sha256(args.checkpoint)!=args.checkpoint_sha256:
        raise ValueError('Trusted frozen checkpoint SHA mismatch')
    with args.checkpoint.open('rb') as stream:
        header=pickle.load(stream)
    if (header.get('format')!=FORMAT or header.get('implementation_sha256')!=CORE_SHA
            or header.get('config',{}).get('phase')!='development' or header['config'].get('seed') not in (0,1)):
        raise ValueError('Require strict v5 development producer; no implicit migration')
    config=RecurrentMatchConfig(**header['config'])
    env=FaultGo1(impl='jax');trainer=RecurrentMatchTrainer(env,config)
    state=trainer.load_checkpoint(args.checkpoint)
    steps=int(state.environment_steps)
    if args.phase=='development' and (steps!=20971520 or int(state.update_index)!=config.num_updates):
        raise ValueError('HPO selection requires its complete precommitted training budget')
    evidence={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':args.checkpoint_sha256,
        'checkpoint_implementation_sha256':header['implementation_sha256'],'checkpoint_format':header['format'],
        'checkpoint_numerical_runtime':header['runtime'],'training_environment_steps':steps,
        'protocol_file_sha256':args.protocol_sha256,'checkpoint_runtime_binding':'strict_producer_runtime',
        'exact_original_paper_score_reproduced':False}
    params,actor=state.params['actor'],trainer.actor
    del state,trainer,header
    evaluate(env=env,actor=actor,config=config,params=params,protocol=protocol,suite=suite,chosen=chosen,
        phase=args.phase,output=args.output,evidence=evidence,replays=args.replays,
        rough_path=args.rough_compatibility,rough_sha=args.rough_compatibility_sha256)
    if file_sha256(args.checkpoint)!=args.checkpoint_sha256:
        raise ValueError('Checkpoint changed during evaluation')


if __name__=='__main__':
    main()
