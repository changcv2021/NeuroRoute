"""Command tracking, survival, and per-case summary metrics."""

def summarize(trace, dt, protocol):
    import numpy as np
    phase = protocol['phase1']
    commands = trace['command']; velocity = trace['actual_velocity'][:,:,:2]
    yaw = trace['actual_yaw_rate']; alive = trace['alive']; rows=[]
    for i,seed in enumerate(phase['evaluation_reset_seeds']):
        command = commands[:,i]; valid = alive[:,i]
        requested = np.linalg.norm(command[:,:2],axis=1)
        actual = np.linalg.norm(velocity[:,i],axis=1)
        eligible = valid & (requested>=.2)
        error2 = np.sum((velocity[:,i]-command[:,:2])**2,axis=1)
        yawerror2 = (yaw[:,i]-command[:,2])**2
        score = np.where(valid,np.exp(-error2/.25-yawerror2/.25),0.)
        stationary = np.exp(-np.sum(command[:,:2]**2,axis=1)/.25-command[:,2]**2/.25)
        rows.append({'reset_seed':seed,'score':float(score.mean()),
            'stationary_score':float(stationary.mean()),'score_difference':float((score-stationary).mean()),
            'mean_command_speed':float(requested[eligible].mean()) if eligible.any() else None,
            'mean_actual_speed':float(actual[eligible].mean()) if eligible.any() else None,
            'speed_ratio':float((actual[eligible]/requested[eligible]).mean()) if eligible.any() else None,
            'near_stationary_fraction':float((actual[eligible]<.05).mean()) if eligible.any() else None,
            'eligible_steps':int(eligible.sum()),'survival_seconds':float(valid.sum()*dt),
            'completed':bool(valid.all()),
            'velocity_rmse':float(np.sqrt(error2[valid].mean())) if valid.any() else None,
            'yaw_rmse':float(np.sqrt(yawerror2[valid].mean())) if valid.any() else None})
    required = ['score','stationary_score','score_difference','mean_command_speed',
                'mean_actual_speed','speed_ratio','near_stationary_fraction','velocity_rmse','yaw_rmse','survival_seconds']
    means = {k:float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None for k in required}
    n_complete = sum(r['completed'] for r in rows)
    bootstrap = np.random.default_rng(619001).choice([r['score_difference'] for r in rows],(5000,len(rows))).mean(axis=1)
    checks = {
        'G1_A':means['speed_ratio'] is not None and means['speed_ratio']>=.6,
        'G1_B':means['near_stationary_fraction'] is not None and means['near_stationary_fraction']<.2,
        'G1_C':means['score']>=.5 and means['score_difference']>=.1,
        'vector_tracking':means['velocity_rmse'] is not None and means['velocity_rmse']<=.35 and means['yaw_rmse']<=.4,
        'survival':n_complete>=7,
        'finite_trajectory':bool(np.all(trace['finite']))}
    return {'case_values':rows,'means':means,'completed_cases':n_complete,
        'termination_rate':1-n_complete/len(rows),'quantitative_checks':checks,
        'quantitative_pass':all(checks.values()),'visual_gate':'pending_inspection',
        'score_difference_case_bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist(),
        'interval_scope':'case uncertainty within one training seed; not training-seed uncertainty'}
