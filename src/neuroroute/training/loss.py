"""PPO with truncated message-state replay and registered routing auxiliaries."""
import dataclasses

import jax
import jax.numpy as jnp
from brax.training.agents.ppo.losses import compute_ppo_loss
from .diagnostics import instrument_loss

observed_ppo_loss = instrument_loss(compute_ppo_loss)


def routing_loss(params, normalizer_params, data, rng, ppo_network, **kwargs):
    obs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data.observation)
    raw = jnp.swapaxes(data.extras['policy_extras']['raw_action'], 0, 1)
    logits = ppo_network.replay(normalizer_params, params.policy, obs, raw[..., :8])
    cached = dataclasses.replace(ppo_network.policy_network, apply=lambda *_: logits)
    network = ppo_network.replace(policy_network=cached)
    total, metrics = observed_ppo_loss(params, normalizer_params, data, rng, network, **kwargs)
    dist = ppo_network.parametric_action_distribution.create_dist(logits)
    up, down = dist.expected_rates()
    target = obs['routing_context'][..., 0]
    dual = jax.lax.stop_gradient(obs['routing_context'][..., 1])
    expected = .5*(up+down)
    penalty = (jnp.mean(dual*(expected-target)) if ppo_network.adaptive_dual
               else 50*jnp.mean((up-target)**2+(down-target)**2))
    directional = jnp.stack((up.mean(), down.mean()))+1e-8
    allocation = directional/directional.sum()
    kl = jnp.sum(allocation*jnp.log(allocation/jnp.asarray(ppo_network.pi_bio)))
    bio = ppo_network.bio_coefficient*kl
    full = total+penalty+bio
    return full, {**metrics, 'total_loss': full, 'communication_budget_penalty': penalty,
        'bio_penalty': bio, 'bio_kl': kl, 'expected_up_fraction':up.mean(),
        'expected_down_fraction':down.mean(), 'expected_total_fraction':expected.mean(),
        'routing/dual_before':dual.mean(), 'routing/budget_target':target.mean(),
        'routing/up_allocation':allocation[0], 'motor_std_mean':dist.scale.mean()}
