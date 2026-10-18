"""Auxiliary diagnostics for the Brax PPO objective."""
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
