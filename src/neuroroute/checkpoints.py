"""Policy checkpoints with preprocessing, configuration, and checksum binding."""
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from .runtime import file_sha256, write_json

def save_checkpoint(path, params, contract, steps):
    """Trusted project-local pickle; bind preprocessing and exact source/config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {'format': 'neuroroute_central_v1', 'contract': contract,
               'environment_steps': int(steps), 'params': jax.device_get(params),
               'scope': 'normalizer_policy_value_not_complete_optimizer_resume'}
    with path.open('xb') as stream:
        pickle.dump(content, stream, protocol=pickle.HIGHEST_PROTOCOL)
    checksum = file_sha256(path)
    write_json(path.with_suffix('.json'), {'sha256': checksum, 'environment_steps': int(steps),
                                          'contract': contract, 'scope': content['scope']})
    return checksum

def load_checkpoint(path, checksum, contract):
    path = Path(path)
    if file_sha256(path) != checksum:
        raise ValueError('Central checkpoint hash mismatch')
    with path.open('rb') as stream:
        content = pickle.load(stream)
    if content['format'] != 'neuroroute_central_v1' or content['contract'] != contract:
        raise ValueError('Central checkpoint source/config/runtime contract mismatch')
    if file_sha256(path) != checksum:
        raise ValueError('Checkpoint changed during load')
    if not all(np.isfinite(x).all() for x in jax.tree.leaves(content['params'])):
        raise FloatingPointError('Nonfinite checkpoint parameters')
    # RunningStatisticsState.std_eps can be a Python float before pmap. Keep
    # scalar metadata unchanged; only stored device-array leaves go back to JAX.
    return jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
                        content['params'])
