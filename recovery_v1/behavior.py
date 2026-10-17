"""Separate diagnostics: unchanged official commands and explicit command challenges."""
import json
from pathlib import Path

from neuroroute.runtime import write_json, file_sha256


def make_rollout(env, network, protocol):
    import jax
    import jax.numpy as jnp
    resets = jnp.asarray(protocol['phase1']['evaluation_reset_seeds'],dtype=jnp.uint32)
    steps = protocol['phase1']['evaluation_steps']
    blocks = jnp.asarray(protocol['phase1']['scripted_command_blocks'],dtype=jnp.float32)
    batch = len(resets)

    def rollout(params, scripted, zero_action):
        states = jax.vmap(env.reset)(jax.vmap(jax.random.PRNGKey)(resets))
        alive = jnp.ones(batch, dtype=jnp.bool_)

        def step(carry, index):
            state, alive = carry
            state = state.replace(info=dict(state.info), metrics=dict(state.metrics), obs=dict(state.obs))
            command = jnp.broadcast_to(blocks[index//250], (batch,3))
            current_command = jnp.where(scripted,command,state.info['command'])
            state.info['command'] = current_command
            # Explicit challenge only: update both command copies and disable natural resampling.
            for name in ('state','privileged_state'):
                state.obs[name] = state.obs[name].at[:,45:48].set(
                    jnp.where(scripted,current_command,state.obs[name][:,45:48]))
            state.info['steps_until_next_cmd'] = jnp.where(scripted,
                jnp.full_like(state.info['steps_until_next_cmd'],100000),state.info['steps_until_next_cmd'])
            observed_command = state.obs['state'][:,45:48]
            logits = network.policy_network.apply(params[0],params[1],state.obs)
            distribution = network.parametric_action_distribution
            raw = distribution.create_dist(logits).loc
            action = jnp.where(zero_action,jnp.zeros_like(raw),distribution.mode(logits))
            target = env._default_pose + action*env._config.action_scale
            previous_q = state.data.qpos[:,7:]
            nxt = jax.vmap(env.step)(state,action)
            velocity = jax.vmap(env.get_local_linvel)(nxt.data)
            yaw = jax.vmap(env.get_gyro)(nxt.data)[:,2]
            finite = jnp.all(jnp.isfinite(nxt.data.qpos),axis=-1) & jnp.all(jnp.isfinite(velocity),axis=-1)
            next_alive = alive & ~nxt.done.astype(bool) & finite
            record = {'command':current_command,'command_in_actor_observation':observed_command,
                'actual_velocity':velocity,'actual_yaw_rate':yaw,
                'policy_raw_mean':raw,'processed_action':action,'requested_joint_target':target,
                'applied_control':nxt.data.ctrl,'actuator_force':nxt.data.actuator_force,
                'joint_position_change':nxt.data.qpos[:,7:]-previous_q,
                'qpos':nxt.data.qpos,'qvel':nxt.data.qvel,'reward':nxt.reward,
                'alive':next_alive,'done':nxt.done,'finite':finite,
                'reward_components':jnp.stack([nxt.metrics['reward/'+k] for k in env._config.reward_config.scales],axis=-1)}
            return (nxt,next_alive),record
        _, traces = jax.lax.scan(step,(states,alive),jnp.arange(steps))
        return traces
    return jax.jit(rollout)


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


def evaluate(rollout, params, env, protocol, output, *, zero_action=False, render=False):
    import jax
    import numpy as np
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    results={}
    for mode in protocol['phase1']['evaluation_modes']:
        trace = jax.device_get(rollout(params,mode=='scripted_commands',zero_action))
        np.savez_compressed(output/f'{mode}.npz',**trace)
        result = summarize(trace,env.dt,protocol)
        result['reward_component_names'] = list(env._config.reward_config.scales)
        result['reward_weighted_components_mean'] = {
            k:float(np.mean(trace['reward_components'][:,:,i]))
            for i,k in enumerate(env._config.reward_config.scales)}
        result['training_reward_mean'] = float(np.mean(trace['reward']))
        result['action_chain']={
            'target_applied_ctrl_max_abs_error':float(np.max(np.abs(trace['requested_joint_target']-trace['applied_control']))),
            'actuator_force_rms':float(np.sqrt(np.mean(trace['actuator_force']**2))),
            'joint_position_change_rms':float(np.sqrt(np.mean(trace['joint_position_change']**2))),
            'raw_action_mean_abs':float(np.mean(np.abs(trace['policy_raw_mean']))),
            'processed_action_mean_abs':float(np.mean(np.abs(trace['processed_action'])))}
        result['source_trace_sha256']=file_sha256(output/f'{mode}.npz')
        results[mode]=result
        if render and mode=='scripted_commands':
            render_trace(env,trace,output)
    result={'kind':'simulated_zero_action' if zero_action else 'trained_official_policy',
        'modes':results,'quantitative_pass':all(v['quantitative_pass'] for v in results.values()),
        'visual_gate':'pending_inspection' if render else 'not_rendered',
        'scope':'development_engineering_gate_not_formal_performance'}
    write_json(output/'result.json',result)
    return result


def render_trace(env,trace,output):
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=mujoco.MjData(env.mj_model)
    renderer=mujoco.Renderer(env.mj_model,height=240,width=320)
    selected=[]
    with imageio.get_writer(output/'scripted_seed41001.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for step in range(0,len(trace['qpos']),2):
            data.qpos[:]=trace['qpos'][step,0]; data.qvel[:]=trace['qvel'][step,0]
            mujoco.mj_forward(env.mj_model,data)
            renderer.update_scene(data,camera='track')
            frame=renderer.render()
            writer.append_data(frame)
            if step in (50,150,300,400,550,650,800,900):
                selected.append((step,frame.copy()))
    renderer.close()
    fig,axes=plt.subplots(4,2,figsize=(8,11),constrained_layout=True)
    for ax,(step,frame) in zip(axes.flat,selected):
        ax.imshow(frame); ax.axis('off')
        ax.set_title(f't={step*.02:.1f}s command={trace["command"][step,0].round(2)}')
    fig.savefig(output/'scripted_contact_sheet.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
    time=np.arange(len(trace['qpos']))*.02
    for i,label in enumerate(['vx / m/s','vy / m/s','yaw / rad/s']):
        actual=trace['actual_velocity'][:,0,i] if i<2 else trace['actual_yaw_rate'][:,0]
        axes[i].plot(time,trace['command'][:,0,i],label='Command')
        axes[i].plot(time,actual,label='Actual');axes[i].set_ylabel(label);axes[i].grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Time / s')
    fig.savefig(output/'scripted_tracking.png',dpi=160);plt.close(fig)
