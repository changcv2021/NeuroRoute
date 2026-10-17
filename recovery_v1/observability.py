"""Auxiliary diagnostics around the unchanged installed Brax PPO objective."""
import functools
import inspect


def instrument_loss(original):
    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo.losses import compute_gae
    signature = inspect.signature(original)

    @functools.wraps(original)
    def observed(*args, **kwargs):
        loss, metrics = original(*args, **kwargs)
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        a = arguments.arguments
        if a['use_distributional_critic']:
            raise ValueError('This audit was specified for the official scalar critic')
        params, norm, network = a['params'], a['normalizer_params'], a['ppo_network']
        data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), a['data'])
        logits = network.policy_network.apply(norm, params.policy, data.observation)
        values = network.value_network.apply(norm, params.value, data.observation)
        terminal = jax.tree.map(lambda x: x[-1], data.next_observation)
        bootstrap = network.value_network.apply(norm, params.value, terminal)
        truncation = data.extras['state_extras']['truncation']
        targets, advantage = compute_gae(truncation, (1-data.discount)*(1-truncation),
            data.reward*a['reward_scaling'], values, bootstrap,
            lambda_=a['gae_lambda'], discount=a['discounting'])
        distribution = network.parametric_action_distribution
        log_prob = distribution.log_prob(logits, data.extras['policy_extras']['raw_action'])
        ratio = jnp.exp(log_prob-data.extras['policy_extras']['log_prob'])
        target_variance = jnp.var(targets)
        diagnostics = {
            'clip_fraction': jnp.mean(jnp.abs(ratio-1)>a['clipping_epsilon']),
            'explained_variance': jnp.where(target_variance>1e-12,
                1-jnp.var(targets-values)/jnp.maximum(target_variance,1e-12), 0.),
            'explained_variance_defined_fraction': (target_variance>1e-12).astype(jnp.float32),
            'raw_advantage_mean': jnp.mean(advantage),
            'raw_advantage_std': jnp.std(advantage),
            'normalized_advantage_mean': jnp.mean((advantage-advantage.mean())/(advantage.std()+1e-8)),
            'normalized_advantage_std': jnp.std((advantage-advantage.mean())/(advantage.std()+1e-8)),
            'entropy': jnp.mean(distribution.entropy(logits, a['rng'])),
        }
        return loss, {**metrics, **jax.tree.map(jax.lax.stop_gradient, diagnostics)}
    return observed


def verify_instrumentation(network, example_observation):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import losses
    keys = jax.random.split(jax.random.PRNGKey(915001), 8)
    normalizer = running_statistics.init_state(example_observation)
    params = losses.PPONetworkParams(policy=network.policy_network.init(keys[0]),
                                    value=network.value_network.init(keys[1]))
    obs = {name: jax.random.normal(jax.random.fold_in(keys[2], i), (8,4)+v.shape)
           for i,(name,v) in enumerate(example_observation.items())}
    logits = network.policy_network.apply(normalizer, params.policy, obs)
    dist = network.parametric_action_distribution
    raw = dist.sample_no_postprocessing(logits, keys[3])
    # Include both clipped and unclipped ratios, nonzero GAE, terminal and timeout samples.
    offsets = jnp.linspace(-.6,.6,32).reshape(8,4)
    data = types.Transition(observation=obs, action=dist.postprocess(raw),
        reward=jax.random.normal(keys[4], (8,4))*.02,
        discount=jnp.ones((8,4)).at[0,-1].set(0.).at[1,-1].set(0.),
        next_observation=jax.tree.map(lambda x: x+.01, obs), extras={
            'state_extras': {'truncation': jnp.zeros((8,4)).at[1,-1].set(1.)},
            'policy_extras': {'raw_action': raw, 'log_prob': dist.log_prob(logits,raw)+offsets,
                              'distribution_params': logits}})
    original = losses.compute_ppo_loss
    observed = instrument_loss(original)
    kwargs = dict(ppo_network=network, entropy_cost=.01, discounting=.97)
    first = jax.jit(jax.value_and_grad(functools.partial(original, **kwargs),has_aux=True))
    second = jax.jit(jax.value_and_grad(functools.partial(observed, **kwargs),has_aux=True))
    (loss_a, metrics_a), grad_a = first(params,normalizer,data,keys[5])
    (loss_b, metrics_b), grad_b = second(params,normalizer,data,keys[5])
    leaves_a, leaves_b = jax.tree.leaves(grad_a), jax.tree.leaves(grad_b)
    gradient_error = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(leaves_a,leaves_b))
    original_metric_error = max(abs(float(metrics_a[k])-float(metrics_b[k])) for k in metrics_a)
    result = {'scalar_loss_absolute_difference': abs(float(loss_a)-float(loss_b)),
              'gradient_max_absolute_difference': gradient_error,
              'original_metric_max_absolute_difference': original_metric_error,
              'diagnostics': {k:float(v) for k,v in metrics_b.items()},
              'predeclared_absolute_tolerance': 1e-6}
    result['passed'] = (max(result[k] for k in ['scalar_loss_absolute_difference',
       'gradient_max_absolute_difference','original_metric_max_absolute_difference']) <= 1e-6
       and all(np.isfinite(float(v)) for v in metrics_b.values())
       and float(metrics_b['clip_fraction'])>0 and float(metrics_b['raw_advantage_std'])>0)
    return result
