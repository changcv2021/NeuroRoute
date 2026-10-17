"""GPU compatibility audit of the actual locked official static rough domain."""
import argparse
import hashlib
import inspect
import os
from pathlib import Path
import time

from .runtime import require_compute,write_json,file_sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--runtime-candidate',choices=['canonical_scatter_workaround','jax082_fast_scatter'],default='canonical_scatter_workaround')
    args=parser.parse_args()
    require_compute(gpu=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .environment import FaultGo1,observation_manifest
    from .reproducible_training import canonical_arrays
    from .development_evaluation import runtime_metadata,evaluation_numerical_runtime
    if args.steps<1 or args.batch_size<1:
        raise ValueError('Positive finite validation length and batch size required')
    started=time.monotonic()
    runtime=evaluation_numerical_runtime(args.runtime_candidate)
    domains={}
    environments={}
    for task in ('flat_terrain','rough_terrain'):
        env=FaultGo1(task=task,impl='jax')
        environments[task]=env
        def run(keys):
            state=canonical_arrays(jax.vmap(env.reset)(keys))
            initial=state
            def step(state,_):
                nxt=canonical_arrays(jax.vmap(env.step)(state,jnp.zeros((args.batch_size,12),jnp.float32)))
                finite=jnp.all(jnp.isfinite(nxt.data.qpos),axis=-1)&jnp.all(jnp.isfinite(nxt.data.qvel),axis=-1)&jnp.all(jnp.isfinite(nxt.obs['state']),axis=-1)
                return nxt,(finite,nxt.done)
            final,trace=jax.lax.scan(step,state,None,length=args.steps)
            return initial,final,trace
        initial,final,(finite,done)=jax.device_get(jax.jit(run)(jax.random.split(jax.random.PRNGKey(0),args.batch_size)))
        model=env.mj_model
        floor=model.geom('floor')
        domains[task]={"observation_shapes":{k:list(v.shape[1:]) for k,v in initial.obs.items()},
            "nq":model.nq,"nv":model.nv,"nu":model.nu,"dt_s":float(env.dt),
            "config":env._config.to_dict(),"observation_schema":observation_manifest(env.leg_map),
            "reward_function_sha256":hashlib.sha256(inspect.getsource(env._get_reward.__func__).encode()).hexdigest(),
            "termination_function_sha256":hashlib.sha256(inspect.getsource(env._get_termination.__func__).encode()).hexdigest(),
            "floor_type":int(floor.type[0]),"floor_friction":floor.friction.tolist(),
            "nhfield":model.nhfield,"hfield_size":model.hfield_size.tolist(),
            "hfield_nrow":model.hfield_nrow.tolist(),"hfield_ncol":model.hfield_ncol.tolist(),
            "hfield_data_sha256":hashlib.sha256(model.hfield_data.tobytes()).hexdigest(),
            "home_qpos":model.keyframe('home').qpos.tolist(),
            "all_finite":bool(np.all(finite)),"termination_steps":int(np.sum(done)),
            "runtime_and_assets":runtime_metadata(env)}
    flat,rough=domains['flat_terrain'],domains['rough_terrain']
    checks={name:flat[name]==rough[name] for name in ['observation_shapes','nq','nv','nu','dt_s','observation_schema','reward_function_sha256','termination_function_sha256']}
    # Verified in the installed MuJoCo make_data dispatch: JAX calls
    # _make_data_jax(m,device); these two options are passed only to WARP.
    # Do not change the environment/reward to make an overstrict audit pass.
    from mujoco.mjx._src import io as mjx_io
    ignored_warp_limits=('naconmax','njmax')
    flat_semantic={k:v for k,v in flat['config'].items() if k not in ignored_warp_limits}
    rough_semantic={k:v for k,v in rough['config'].items() if k not in ignored_warp_limits}
    checks['both_jax_backend']=flat['config']['impl']==rough['config']['impl']=='jax'
    checks['task_config_except_unused_warp_limits']=flat_semantic==rough_semantic
    checks['both_all_finite']=flat['all_finite'] and rough['all_finite']
    checks['zero_action_smoke_no_early_fall']=flat['termination_steps']==0 and rough['termination_steps']==0
    checks['actual_static_heightfield']=flat['nhfield']==0 and rough['nhfield']==1
    result={"status":"compatible_fixed_official_rough" if all(checks.values()) else "blocked_incompatible_rough",
        "checks":checks,"domains":domains,"numerical_runtime":runtime,
        "runtime_candidate":args.runtime_candidate,"validator_version":"rough-compatibility-v3-explicit-runtime",
        "evaluator_support_sha256":file_sha256(Path(__file__).with_name('development_evaluation.py')),
        "job_id":os.environ['SLURM_JOB_ID'],"validator_sha256":file_sha256(__file__),
        "steps":args.steps,"batch_size":args.batch_size,"elapsed_s":time.monotonic()-started,
        "terrain_definition":"single locked official heightfield;100 reset/command/noise scripts; not100 terrains",
        "terrain_seed_role":"reserved_unused","remaining_scope":"policy evaluation and domain performance have not been tested",
        "configuration_difference_audit":{"ignored_only_for_jax":{k:{"flat":flat['config'][k],"rough":rough['config'][k]} for k in ignored_warp_limits},
            "source":str(mjx_io.__file__),"source_sha256":file_sha256(mjx_io.__file__),
            "reason":"make_data JAX dispatch ignores these two allocation limits; WARP dispatch consumes them"},
        "domain_changes":["heightfield instead of plane","floor friction1.0 instead of0.6","home height0.35 instead of0.278"]}
    write_json(args.output,result)
    print(result['status'],checks,flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


if __name__=='__main__':
    main()
