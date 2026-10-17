"""E1/E2 reporting: independent training seeds, shared case-script contrasts.

This module does not execute a policy, select a checkpoint, or alter metrics.
"""
import math
import os
import socket

from .metrics import hierarchical_bootstrap,retention

FORMAL_SEEDS=(11,22,33,44,55)
PRIMARY=(('healthy','score'),('actuator','postevent_deficit_area_s'),
         ('sensor_freeze','postevent_deficit_area_s'))


def checked_number(value,*,bounded=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Expected a finite measured number, not missing data')
    if bounded and not 0<=value<=1:
        raise ValueError('Task score must be bounded')
    return float(value)


def flatten_case(record,case,run,*,weak_denominator=1e-6,healthy=None):
    """Preserve measured failures as zero scores; absent measurements stay absent."""
    tracking=record['tracking'];comm=record['communication'];recovery=record.get('recovery') or {}
    score=checked_number(tracking['score'],bounded=True)
    central=run['method']=='B_CENTRAL'
    bits=comm.get('active_bits_per_step')
    if central:
        if bits is not None or comm.get('status')!='not_applicable':
            raise ValueError('Central cost must be explicitly N/A')
    else:
        bits=checked_number(bits)
        if not 8<=bits<=520:
            raise ValueError('Wire cost outside the audited flag/payload interface')
    fraction=run.get('budget_fraction')
    target=None if central or fraction is None else 520*fraction
    fault=case.get('fault') or {}
    ratio=None;difference=None;healthy_score=None;retention_status='not_a_fault_case'
    if case.get('fault') or case['kind']=='rough':
        retention_status='healthy_pair_missing'
        if healthy is not None:
            paired=retention(score,healthy,minimum_healthy_score=weak_denominator)
            ratio=paired['ratio'];difference=paired['difference'];healthy_score=healthy
            retention_status='weak_healthy_denominator' if ratio is None else 'available'
    result={'run_id':run['run_id'],'method':run['method'],'budget_fraction':fraction,
        'phase':record.get('phase'),'formal_result':record.get('formal_result',False),
        'training_seed':run['seed'],'case_id':case['case_id'],'case_sha256':case['case_sha256'],
        'script_id':case['script_id'],'kind':case['kind'],'score':score,
        'velocity_rmse_m_s':tracking.get('velocity_rmse_m_s'),'yaw_rmse_rad_s':tracking.get('yaw_rmse_rad_s'),
        'physical_metric_observed_steps':tracking.get('physical_metric_observed_steps'),
        'completed':bool(tracking['completed']),'survival_s':tracking.get('survival_s'),
        'failure_step':tracking.get('failure_step'),'nonfinite_failure':bool(record.get('nonfinite_failure',False)),
        'active_bits_per_step':bits,'active_bits_per_s':comm.get('active_bits_per_s'),
        'sent_total_bits':comm.get('sent_total_bits'),'received_total_bits':comm.get('received_total_bits'),
        'upward_payload_fraction':comm.get('upward_payload_fraction'),
        'upward_sent_bits':comm.get('upward_sent_bits'),'downward_sent_bits':comm.get('downward_sent_bits'),
        'active_steps':comm.get('active_steps'),'budget_target_bits_per_step':target,
        'budget_excess_bits_per_step':None if target is None else max(0.,bits-target),
        'episode_budget_feasible':None if target is None else bits<=target,
        'communication_status':'not_applicable' if central else 'measured',
        'leg':fault.get('leg'),'loss_fraction':fault.get('loss_fraction'),
        'freeze_duration_s':fault.get('duration_s'),'fault_onset_step':fault.get('onset_step'),
        'postevent_deficit_area_s':recovery.get('postevent_deficit_area_s'),
        'postevent_horizon_s':recovery.get('postevent_horizon_s'),
        'recovery_time_s':recovery.get('recovery_time_s'),'recovery_status':recovery.get('status'),
        'right_censored':recovery.get('right_censored'),'censor_time_s':recovery.get('censor_time_s'),
        'failed_before_event':recovery.get('failed_before_event'),
        'fault_applied_before_failure':recovery.get('fault_applied_before_failure'),
        'healthy_pair_score':healthy_score,'retention_difference':difference,'retention_ratio':ratio,
        'retention_status':retention_status}
    for prefix,field in (('step_cost','bits_per_step_quantiles'),('cache_age_s','cache_age_s_quantiles')):
        for q in ('0.5','0.9','0.99'):
            result[f'{prefix}_episode_quantile_{q}']=(comm.get(field) or {}).get(q)
    for key in ('postevent_deficit_area_s','postevent_horizon_s','recovery_time_s'):
        if result[key] is not None and checked_number(result[key])<0:
            raise ValueError('Negative recovery duration/area')
    if case.get('fault') and result['postevent_deficit_area_s'] is None:
        raise ValueError('Fault evaluation lacks its failure-aware deficit area')
    return result


def paired_case_contrast(left,right,*,replicates=5000,seed=20260912):
    """left/right: seed -> {case identity: metric}; positive means left minus right.

    Training-seed draws are independent between methods. Only the inner case
    indices are shared. Equal numeric seed labels never pair neural weights.
    """
    import numpy as np
    if replicates>100 and (not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login')):
        raise RuntimeError('Production bootstrap requires a compute allocation')
    if not left or not right or replicates<1:
        raise ValueError('Both methods require independent training-seed samples')
    case_keys=set(next(iter(left.values())))
    if not case_keys or any(set(v)!=case_keys for group in (left,right) for v in group.values()):
        raise ValueError('Paired contrast requires exactly shared case identities; do not intersect away missing cases')
    keys=sorted(case_keys)
    a=np.asarray([[checked_number(v[k]) for k in keys] for _,v in sorted(left.items())])
    b=np.asarray([[checked_number(v[k]) for k in keys] for _,v in sorted(right.items())])
    rng=np.random.Generator(np.random.PCG64(seed));samples=np.empty(replicates)
    for i in range(replicates):
        ai=rng.integers(len(a),size=len(a));bi=rng.integers(len(b),size=len(b))
        ci=rng.integers(len(keys),size=len(keys))
        samples[i]=a[ai][:,ci].mean()-b[bi][:,ci].mean()
    return {'difference':float(a.mean()-b.mean()),'interval':np.quantile(samples,[.025,.975]).tolist(),
        'confidence':.95,'replicates':replicates,'bootstrap_seed':seed,
        'left_seed_count':len(a),'right_seed_count':len(b),'shared_case_count':len(keys),
        'left_per_seed':{str(k):float(np.mean(list(v.values()))) for k,v in left.items()},
        'right_per_seed':{str(k):float(np.mean(list(v.values()))) for k,v in right.items()},
        'pairing':'shared exogenous cases only; independent outer training-seed draws',
        'sign':'left minus right','interval_family':'pointwise; not familywise simultaneous'}


def summarize(rows,runs,coverage,*,replicates=5000,seed=20260912):
    """Incomplete runs retain rows/coverage but never silently enter main CIs."""
    import numpy as np
    complete={r['run_id'] for r in coverage if r['status']=='complete'}
    coverage_by_id={r['run_id']:r for r in coverage}
    per_seed=[];groups={};known={(r['method'],r.get('budget_fraction')) for r in runs}
    for method,budget in sorted(known,key=str):
        expected=[r for r in runs if (r['method'],r.get('budget_fraction'))==(method,budget)]
        for kind in ('healthy','actuator','sensor_freeze','rough'):
            selected=[r for r in rows if r['method']==method and r['budget_fraction']==budget and r['kind']==kind and r['run_id'] in complete]
            stats={};metrics=('score','postevent_deficit_area_s','retention_difference','retention_ratio',
                                'active_bits_per_step','velocity_rmse_m_s','yaw_rmse_rad_s','completed','survival_s')
            for metric in metrics:
                data={}
                for run in expected:
                    observed=[r for r in selected if r['run_id']==run['run_id']]
                    values=[r[metric] for r in observed if r[metric] is not None]
                    per_seed.append({'run_id':run['run_id'],'method':method,'budget_fraction':budget,
                        'training_seed':run['seed'],'kind':kind,'metric':metric,
                        'value':float(np.mean(values)) if values else None,'measured_cases':len(values),
                        'missing_or_undefined_cases':coverage_by_id[run['run_id']].get('expected_kind_counts',{}).get(kind,len(observed))-len(values),
                        'run_complete':run['run_id'] in complete})
                    if values:data[run['seed']]=values
                if data:
                    stats[metric]=hierarchical_bootstrap(data,replicates=replicates,seed=seed)
                    stats[metric]['complete_registered_seed_coverage']=len(data)==len(expected)
                    if metric=='retention_ratio':stats[metric]['interpretation']='conditional on healthy denominator above threshold; undefined ratios excluded and counted'
            wire=[r for r in selected if r['active_bits_per_step'] is not None]
            seed_costs=[float(np.mean([r['active_bits_per_step'] for r in wire if r['training_seed']==s])) for s in sorted({r['training_seed'] for r in wire})]
            target=None if method=='B_CENTRAL' or budget is None else 520*budget
            budget_status=('not_applicable' if target is None else 'missing' if not seed_costs else
                'infeasible' if max(seed_costs)>target else 'feasible_measured_seeds')
            groups[f'{method}:{budget}:{kind}']={'method':method,'budget_fraction':budget,'kind':kind,
                'expected_training_seeds':[r['seed'] for r in expected],
                'measured_training_seeds':sorted({r['training_seed'] for r in selected}),
                'statistics':stats,'budget_status':budget_status,'budget_target_bits_per_step':target,
                'maximum_seed_mean_budget_excess':None if target is None or not seed_costs else max(0.,max(seed_costs)-target),
                'episode_budget_exceedance_fraction':float(np.mean([not r['episode_budget_feasible'] for r in wire])) if wire and target else None,
                'recovery_status_counts':{status:sum(r['recovery_status']==status for r in selected) for status in
                    ('recovered','failed_before_event','failed_after_event','unrecovered_at_horizon')},
                'retention_undefined_count':sum(r['retention_status'] in ('weak_healthy_denominator','healthy_pair_missing') for r in selected)}
    contrasts=[]
    for kind,metric in PRIMARY:
        populations=[];missing=[]
        for method in ('M_BIO','B_ASYM'):
            registered=[r for r in runs if r['method']==method and r.get('budget_fraction')==.25]
            if not registered:missing.append(method+':unregistered')
            missing.extend(r['run_id'] for r in registered if r['run_id'] not in complete)
            population={r['seed']:{x['case_sha256']:x[metric] for x in rows if x['run_id']==r['run_id'] and x['kind']==kind}
                        for r in registered if r['run_id'] in complete}
            populations.append(population)
        result={'left':'M_BIO','right':'B_ASYM','budget_fraction':.25,'kind':kind,'metric':metric,
                'primary':True,'higher_is_better':metric=='score'}
        if missing or not all(populations):result.update(status='missing_registered_runs',missing=missing)
        else:result.update(status='complete',**paired_case_contrast(*populations,replicates=replicates,seed=seed))
        contrasts.append(result)
    return per_seed,groups,contrasts
