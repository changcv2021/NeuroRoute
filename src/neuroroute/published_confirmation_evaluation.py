"""Strict version3 MATCH producer adapter for common formal/validation cases."""
import argparse
import json
from pathlib import Path
import pickle

from .runtime import require_compute,file_sha256
from .confirmation_evaluation import checked_suite,selected_cases
from .published_confirmation_protocol import (PublishedConfirmationConfig,
    CONFIRMATION_SEEDS,VALIDATION_SEEDS)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--phase',choices=['confirmation','validation'],required=True)
    p.add_argument('--evaluation-protocol',type=Path,required=True);p.add_argument('--evaluation-protocol-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--replays',type=int,default=1)
    p.add_argument('--limit',type=int)
    p.add_argument('--rough-compatibility',type=Path);p.add_argument('--rough-compatibility-sha256')
    args=p.parse_args()
    if file_sha256(args.evaluation_protocol)!=args.evaluation_protocol_sha256:
        raise ValueError('Common formal evaluation protocol changed')
    protocol=json.loads(args.evaluation_protocol.read_text())
    suite=checked_suite(protocol,args.phase);chosen=selected_cases(suite,args.phase,limit=args.limit)
    require_compute(gpu=True)
    from .environment import FaultGo1
    from .published_confirmation_training import PublishedConfirmationTrainer
    from .recurrent_evaluation import evaluate,CORE_SHA
    if file_sha256(args.checkpoint)!=args.checkpoint_sha256:
        raise ValueError('Trusted frozen checkpoint SHA mismatch')
    with args.checkpoint.open('rb') as stream:header=pickle.load(stream)
    if any(header.get(k)!=v for k,v in {'format_version':3,'checkpoint_kind':'neuroroute_confirmation_v1',
        'phase':args.phase,'method':'B_PUBLISHED','algorithm_implementation_sha256':CORE_SHA,
        'normalizer':{'kind':'fixed_identity','updates':0}}.items()):
        raise ValueError('Require the exact phase and accepted v5 recursive version3 producer')
    seeds=CONFIRMATION_SEEDS if args.phase=='confirmation' else VALIDATION_SEEDS
    if header.get('training_seed') not in seeds:
        raise ValueError('Producer training seed belongs to another phase')
    config=PublishedConfirmationConfig(**header['config']);config.validate()
    resolved,_=config.validate_manifests()
    formal=args.phase=='confirmation'
    if formal and resolved.get('evaluation_protocol_sha256')!=args.evaluation_protocol_sha256:
        raise ValueError('Training protocol is not pre-bound to the common formal600')
    env=FaultGo1(impl='jax');trainer=PublishedConfirmationTrainer(env,config)
    state=trainer.load_checkpoint(args.checkpoint)
    steps=int(state.environment_steps);target=config.num_updates*config.num_envs*config.horizon
    if formal and (steps!=200015872 or steps!=target):
        raise ValueError('Formal evaluation requires the completed common training step budget')
    evidence={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':args.checkpoint_sha256,
        'checkpoint_format_version':3,'checkpoint_kind':header['checkpoint_kind'],
        'checkpoint_implementation_sha256':header['implementation_sha256'],
        'checkpoint_numerical_runtime':header['numerical_runtime'],
        'training_environment_steps':steps,'training_environment_steps_target':target,
        'protocol_sha256':config.protocol_sha256,'selection_sha256':config.selection_sha256,
        'resolved_protocol_sha256':config.protocol_sha256,'selected_config_sha256':config.selection_sha256,
        'evaluation_protocol_file_sha256':args.evaluation_protocol_sha256,
        'producer_adapter_sha256':file_sha256(__file__),'method_key':config.method_key,
        'exact_original_paper_score_reproduced':False}
    params,actor=state.params['actor'],trainer.actor
    del state,trainer,header
    evaluate(env=env,actor=actor,config=config,params=params,protocol=protocol,suite=suite,chosen=chosen,
        phase=args.phase,output=args.output,evidence=evidence,replays=args.replays,
        rough_path=args.rough_compatibility,rough_sha=args.rough_compatibility_sha256)
    if file_sha256(args.checkpoint)!=args.checkpoint_sha256:
        raise ValueError('Checkpoint changed during evaluation')


if __name__=='__main__':main()
