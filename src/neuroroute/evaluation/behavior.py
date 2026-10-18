"""Separate diagnostics: unchanged official commands and explicit command challenges."""
import json
from pathlib import Path

from neuroroute.runtime import write_json, file_sha256


def make_rollout(env, network, protocol, diagnostic_apply):
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
            key = jax.random.fold_in(jax.random.PRNGKey(610001),index)
            raw = distribution.create_dist(logits).sample(key,motor_noise=False)
            raw = jnp.where(zero_action,jnp.zeros_like(raw),raw)
            action = distribution.postprocess(raw)
            _, route = diagnostic_apply(params[0],params[1],state.obs,raw[:,:4],raw[:,4:8])
            state = state.replace(obs={**state.obs, 'communication':route['next_communication_state']})
            target = env._default_pose + action[:,8:]*env._config.action_scale
            previous_q = state.data.qpos[:,7:]
            nxt = jax.vmap(env.step)(state,action)
            velocity = jax.vmap(env.get_local_linvel)(nxt.data)
            yaw = jax.vmap(env.get_gyro)(nxt.data)[:,2]
            finite = jnp.all(jnp.isfinite(nxt.data.qpos),axis=-1) & jnp.all(jnp.isfinite(velocity),axis=-1)
            next_alive = alive & ~nxt.done.astype(bool) & finite
            record = {'command':current_command,'command_in_actor_observation':observed_command,
                'actual_velocity':velocity,'actual_yaw_rate':yaw,
                'policy_raw_mean':raw[:,8:],'processed_action':action[:,8:],'requested_joint_target':target,
                'applied_control':nxt.data.ctrl,'actuator_force':nxt.data.actuator_force,
                'joint_position_change':nxt.data.qpos[:,7:]-previous_q,
                'qpos':nxt.data.qpos,'qvel':nxt.data.qvel,'reward':nxt.reward,
                'alive':next_alive,'done':nxt.done,'finite':finite,
                'communication_cache':state.obs['communication'],
                'ascending_router_logits':route['ascending_router_logits'], 'descending_router_logits':route['descending_router_logits'],
                'ascending_payload':route['ascending_payload'], 'descending_payload':route['descending_payload'],
                'ascending_send':route['up_wire']['send'], 'descending_send':route['down_wire']['send'],
                'ascending_payload_bits':route['up_wire']['payload_bits'], 'descending_payload_bits':route['down_wire']['payload_bits'],
                'ascending_overhead_bits':route['up_wire']['overhead_bits'], 'descending_overhead_bits':route['down_wire']['overhead_bits'],
                'reward_components':jnp.stack([nxt.metrics['reward/'+k] for k in env._config.reward_config.scales],axis=-1)}
            return (nxt,next_alive),record
        _, traces = jax.lax.scan(step,(states,alive),jnp.arange(steps))
        return traces
    return jax.jit(rollout)



from .metrics import summarize
from .rendering import render_trace


def evaluate(rollout, params, env, protocol, output, *, zero_action=False, render=False):
    import jax
    import numpy as np
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    results={}
    for mode in protocol['phase1']['evaluation_modes']:
        trace=jax.device_get(rollout(params,mode=='scripted_commands',zero_action))
        np.savez_compressed(output/f'{mode}.npz',**trace)
        result=summarize(trace,env.dt,protocol)
        result['reward_component_names']=list(env._config.reward_config.scales)
        result['reward_weighted_components_mean']={k:float(np.mean(trace['reward_components'][:,:,i]))
            for i,k in enumerate(env._config.reward_config.scales)}
        result['training_reward_mean']=float(np.mean(trace['reward']))
        result['action_chain']={
            'target_applied_ctrl_max_abs_error':float(np.max(np.abs(trace['requested_joint_target']-trace['applied_control']))),
            'actuator_force_rms':float(np.sqrt(np.mean(trace['actuator_force']**2))),
            'joint_position_change_rms':float(np.sqrt(np.mean(trace['joint_position_change']**2))),
            'raw_action_mean_abs':float(np.mean(np.abs(trace['policy_raw_mean']))),
            'processed_action_mean_abs':float(np.mean(np.abs(trace['processed_action'])))}
        wire={};valid=True
        for direction in ('ascending','descending'):
            payload=trace[direction+'_payload']; flags=trace[direction+'_send']
            pb=trace[direction+'_payload_bits'];ob=trace[direction+'_overhead_bits']
            expected=flags.astype(np.int64)*(32*payload.shape[-1])
            okay=bool(np.isfinite(payload).all() and np.all(payload[~flags]==0) and np.array_equal(pb,expected)
                      and np.array_equal(ob,flags.astype(np.int64)*4))
            valid=valid and okay
            wire[direction]={'packets':int(flags.sum()),'payload_bits':int(pb.sum()),
                'overhead_bits':int(ob.sum()),'bits_per_control_step':float((pb+ob).sum(axis=-1).mean()),
                'payload_dimension':payload.shape[-1],'payload_dtype':str(payload.dtype)}
        wire['total_bits_per_control_step']=sum(wire[d]['bits_per_control_step'] for d in ('ascending','descending'))
        wire['sparse_wire_pass']=valid
        wire['full_wire_pass']=valid and wire['total_bits_per_control_step']==8224
        wire['scope']='Logical transmitted interface bits over all evaluated steps, including post-fall diagnostic continuation; not hardware network measurements'
        result['communication']=wire
        result['router_diagnostics']={'mode':'sampled_gates_deterministic_motor','finite':bool(np.isfinite(trace['ascending_router_logits']).all() and np.isfinite(trace['descending_router_logits']).all()),
            'ascending_logit_mean':float(trace['ascending_router_logits'].mean()),'descending_logit_mean':float(trace['descending_router_logits'].mean()),
            'scope':'FlyCNS Bernoulli gates; Gaussian motor location conditional on actual gates'}
        result['source_trace_sha256']=file_sha256(output/f'{mode}.npz')
        results[mode]=result
        if render and mode=='scripted_commands':render_trace(env,trace,output)
    result={'kind':'simulated_zero_action' if zero_action else 'flycns_v2_policy',
        'modes':results,'quantitative_pass':all(v['quantitative_pass'] for v in results.values()),
        'visual_gate':'pending_inspection' if render else 'not_rendered',
        'scope':'development_engineering_gate_not_formal_performance'}
    write_json(output/'result.json',result)
    return result
