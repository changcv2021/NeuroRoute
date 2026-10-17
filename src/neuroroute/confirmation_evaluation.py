"""Independent strict v3 evaluation; validation never becomes confirmation data."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time

from .evaluation_cases import build_suite,digest,keyed_seed,key_words,write_immutable
from .runtime import file_sha256,require_compute,write_json

SHARED_SOURCE_SHA256={
    'development_evaluation.py':'bf9ad99f1a01be0523601848ca379a27da4a851a78e29db413829e5772b80339',
    'evaluation_cases.py':'b482cfc063346c96c695ee81c85b717b4ffe14f2cfb5a713ef8265f32996dc86',
    'metrics.py':'bee8c26774a70665438cd5dccf8c3d32225c85f263ce32a8e5033793b5a90c54'}


def checked_suite(protocol,phase):
    """Preserve original test cases; technical checks get a separate namespace."""
    if protocol.get('status')!='formal_frozen':
        raise ValueError('Require the explicit formal-frozen evaluation protocol copy')
    if phase=='confirmation':
        suite=build_suite(protocol,'test')
    elif phase=='validation':
        technical=dict(protocol)
        technical['development_namespace']='neuroroute.validation.eval.v3.static_rough'
        suite=build_suite(technical,'development')
        for case in suite['cases']:
            case['case_id']=case['case_id'].replace('development:','validation:',1)
            case['case_sha256']=digest({k:v for k,v in case.items() if k!='case_sha256'})
        suite['cases_sha256']=digest(suite['cases'])
        suite['split']='validation'
        suite['technical_validation']=True
    else:raise ValueError('Explicit confirmation or technical validation phase required')
    if len(suite['cases'])!=600 or {k:sum(c['kind']==k for c in suite['cases']) for k in protocol['counts']}!={'healthy':100,'actuator':200,'sensor_freeze':200,'rough':100}:
        raise ValueError('Expected complete 600-case evaluation design')
    return suite


def phase_gate(header,phase):
    from .confirmation_protocol import CONFIRMATION_SEEDS,VALIDATION_SEEDS
    if (header.get('format_version')!=3 or header.get('checkpoint_kind')!='neuroroute_confirmation_v1'
            or header.get('phase')!=phase or header.get('config',{}).get('phase')!=phase):
        raise ValueError('Version3 producer phase mismatch; development/validation cannot become formal')
    seeds=CONFIRMATION_SEEDS if phase=='confirmation' else VALIDATION_SEEDS if phase=='validation' else ()
    if type(header.get('training_seed')) is not int or header['training_seed'] not in seeds:
        raise ValueError('Checkpoint training seed belongs to another phase')
    if header.get('method')=='B_PUBLISHED':
        raise ValueError('Unaccepted MATCH interface is not supported by this evaluator')
    if header.get('normalizer')!={'kind':'fixed_identity','updates':0}:
        raise ValueError('Only the frozen identity normalizer is supported')


def selected_cases(suite,phase,case_ids=None,limit=None):
    if phase=='confirmation' and (case_ids or limit is not None):
        raise ValueError('Formal entry evaluates all600 cases; no test-based subset selection')
    if case_ids:
        wanted=set(case_ids);selected=[c for c in suite['cases'] if c['case_id'] in wanted]
        if {c['case_id'] for c in selected}!=wanted:raise ValueError('Unknown technical-validation case ID')
    else:selected=suite['cases']
    if limit is not None:
        if limit<1:raise ValueError('Positive technical-validation limit required')
        selected=selected[:limit]
    return selected


def phase_case_arrays(cases,method,seed,phase):
    import numpy as np
    from .development_evaluation import case_arrays
    arrays=case_arrays(cases,method,seed)
    namespace=f'neuroroute.{phase}.policy.v1:{method}:{seed}'
    arrays['policy_keys']=np.asarray([[key_words(keyed_seed(namespace,c['script_id'],'communication',step))
        for c in cases] for step in range(cases[0]['horizon_steps'])],np.uint32)
    arrays['policy_namespace']=namespace
    return arrays


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--phase',choices=['confirmation','validation'],required=True)
    p.add_argument('--evaluation-protocol',type=Path,required=True);p.add_argument('--evaluation-protocol-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--case-id',action='append');p.add_argument('--limit',type=int)
    p.add_argument('--replays',type=int,default=1);p.add_argument('--save-state-trace',action='store_true')
    p.add_argument('--rough-compatibility',type=Path);p.add_argument('--rough-compatibility-sha256')
    args=p.parse_args()
    if args.batch_size<1 or args.replays<1:raise ValueError('Positive batch size/replays required')
    for name,sha in SHARED_SOURCE_SHA256.items():
        if file_sha256(Path(__file__).with_name(name))!=sha:raise ValueError(f'Frozen shared evaluator source changed: {name}')
    if file_sha256(args.evaluation_protocol)!=args.evaluation_protocol_sha256:raise ValueError('Evaluation protocol SHA mismatch')
    protocol=json.loads(args.evaluation_protocol.read_text())
    suite=checked_suite(protocol,args.phase);chosen=selected_cases(suite,args.phase,args.case_id,args.limit)
    require_compute(gpu=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .confirmation_protocol import ConfirmationConfig
    from .confirmation_training import make_confirmation_trainer
    from .environment import FaultGo1,observation_manifest
    from .development_evaluation import make_rollout,summarize_case,runtime_metadata,compare_replay,validate_rough_domain
    checkpoint_hash=file_sha256(args.checkpoint)
    if checkpoint_hash!=args.checkpoint_sha256:raise ValueError('Frozen checkpoint SHA mismatch')
    with args.checkpoint.open('rb') as stream:header=pickle.load(stream)
    phase_gate(header,args.phase)
    config=ConfirmationConfig(**header['config']);config.validate()
    resolved,selection=config.validate_manifests()
    formal=args.phase=='confirmation'
    if formal and resolved.get('evaluation_protocol_sha256')!=args.evaluation_protocol_sha256:
        raise ValueError('Resolved training protocol must pre-bind this exact evaluation protocol')
    env=FaultGo1(impl='jax')
    trainer=make_confirmation_trainer(env,config)
    state=trainer.load_checkpoint(args.checkpoint)
    header.pop('state',None)
    if file_sha256(args.checkpoint)!=checkpoint_hash:raise ValueError('Checkpoint changed during producer validation')
    steps=int(state.environment_steps);target=header['training_environment_steps_target']
    if formal and steps!=target:raise ValueError('Formal evaluation requires the final resolved training step')
    if float(env.dt)!=protocol['control_dt_s']:raise ValueError('Evaluation control dt differs from checkpoint environment')
    params=state.params['actor'];actor=trainer.actor
    del state,trainer
    rough_env=None;rough_domain=None
    if any(c['kind']=='rough' for c in chosen):
        if not args.rough_compatibility or not args.rough_compatibility_sha256:
            raise ValueError('All requested rough cases require the actual audited Fast domain')
        rough_env=FaultGo1(task='rough_terrain',impl='jax')
        rough_domain=validate_rough_domain(protocol,args.rough_compatibility,args.rough_compatibility_sha256,rough_env,'jax082_fast_scatter')
    args.output.mkdir(parents=True,exist_ok=False)
    write_immutable(args.output/'evaluation_suite.json',suite)
    write_json(args.output/'training_config.json',asdict(config))
    write_json(args.output/'observation_schema.json',observation_manifest(env.leg_map))
    provenance={'phase':args.phase,'formal_result':formal,'checkpoint_format_version':3,
        'checkpoint_kind':header['checkpoint_kind'],'method':config.method,'method_key':config.method_key,'training_seed':config.seed,
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_hash,'training_environment_steps':steps,
        'training_environment_steps_target':target,'protocol_sha256':config.protocol_sha256,'selection_sha256':config.selection_sha256,
        'resolved_protocol_sha256':config.protocol_sha256,'selected_config_sha256':config.selection_sha256,
        'checkpoint_implementation_sha256':header['implementation_sha256'],'checkpoint_numerical_runtime':header['numerical_runtime'],
        'runtime_candidate':'jax082_fast_scatter','checkpoint_family':'central' if config.method=='B_CENTRAL' else 'mixed',
        'evaluator_version':'confirmation-evaluator-v1','evaluator_sha256':file_sha256(__file__),'shared_source_sha256':SHARED_SOURCE_SHA256,
        'evaluation_protocol_file_sha256':args.evaluation_protocol_sha256,'protocol_canonical_sha256':digest(protocol),
        'suite_cases_sha256':suite['cases_sha256'],'selected_cases_sha256':digest(chosen),
        'selected_case_ids':[c['case_id'] for c in chosen],'batch_size':args.batch_size,'replays':args.replays,
        'normalization_updates':0,'command_schedule_override':True,'canonical_strong_arrays':True,
        'runtime':runtime_metadata(env),'environment_config':env._config.to_dict(),'rough_domain':rough_domain,
        'job_id':os.environ['SLURM_JOB_ID'],'save_state_trace':args.save_state_trace,
        'validation_partial_checkpoint':not formal and steps!=target,
        'noise':'absolute-step exogenous keys; independent phase/method/training-seed policy namespace'}
    write_json(args.output/'provenance.json',provenance)
    started=time.monotonic();records=[];comparisons=[];offset=0
    with (args.output/'cases.jsonl').open('x') as stream:
        for domain,group in [(env,[c for c in chosen if c['kind']!='rough']),(rough_env,[c for c in chosen if c['kind']=='rough'])]:
            if not group:continue
            rollout=make_rollout(domain,actor,config,group[0]['horizon_steps'],args.save_state_trace,True)
            for start in range(0,len(group),args.batch_size):
                batch=group[start:start+args.batch_size];padded=batch+[batch[-1]]*(args.batch_size-len(batch))
                arrays=phase_case_arrays(padded,config.method,config.seed,args.phase);namespace=arrays.pop('policy_namespace')
                arrays=jax.tree.map(jnp.asarray,arrays)
                trace={k:v[:,:len(batch)] for k,v in jax.device_get(rollout(params,arrays)).items()}
                path=args.output/f'trace_{offset:05d}.npz';np.savez_compressed(path,**trace)
                for i,case in enumerate(batch):
                    result=summarize_case(case,{k:v[:,i] for k,v in trace.items()},protocol,config)
                    # The common metric helper is development-labeled. These
                    # labels are assigned only after strict producer validation.
                    result.update(phase=args.phase,formal_result=formal,trace_file=path.name,trace_batch_index=i,
                        trace_sha256=file_sha256(path),policy_rng_namespace=namespace)
                    if case['kind']=='rough':result['rough_domain']=rough_domain
                    if result['max_actor_command_error']>1e-6 or (result['max_sensor_freeze_error'] is not None and result['max_sensor_freeze_error']>1e-6):
                        raise AssertionError('Command/freeze actor-observation contract violated')
                    records.append(result);stream.write(json.dumps(result,allow_nan=False)+'\n')
                for repeat in range(1,args.replays):
                    replay={k:v[:,:len(batch)] for k,v in jax.device_get(rollout(params,arrays)).items()}
                    comparison=compare_replay(trace,replay);comparison.update(batch_offset=offset,repetition=repeat,independent_trial=False)
                    comparisons.append(comparison)
                offset+=len(batch);stream.flush()
                write_json(args.output/'progress.json',{'phase':args.phase,'formal_result':formal,'completed_cases':offset,'requested_cases':len(chosen)})
                print(json.dumps({'completed_cases':offset,'elapsed_s':time.monotonic()-started}),flush=True)
    by_kind={}
    for kind in ('healthy','actuator','sensor_freeze','rough'):
        group=[r for r in records if r['kind']==kind]
        if group:
            by_kind[kind]={'n':len(group),'mean_score':float(np.mean([r['tracking']['score'] for r in group])),
                'mean_active_bits_per_step':None if config.method=='B_CENTRAL' else float(np.mean([r['communication']['active_bits_per_step'] for r in group])),
                'nonfinite_failures':sum(r['nonfinite_failure'] for r in group)}
    write_json(args.output/'replay_diagnostics.json',{'comparisons':comparisons,'independent_additional_trials':0})
    summary={**provenance,'status':'completed_confirmation_evaluation' if formal else 'completed_validation_evaluation',
        'completed_cases':len(records),'blocked_cases':0,'by_kind':by_kind,'elapsed_s':time.monotonic()-started,
        'cases_jsonl_sha256':file_sha256(args.output/'cases.jsonl'),'replay_diagnostics_sha256':file_sha256(args.output/'replay_diagnostics.json')}
    write_json(args.output/'summary.json',summary)


if __name__=='__main__':main()
