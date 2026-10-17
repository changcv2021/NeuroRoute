"""Strict confirmation identity around unchanged v5 recurrent PPO mathematics."""
import hashlib
from pathlib import Path
import numpy as np

from .confirmation_training import ConfirmationCheckpointMixin
from .published_confirmation_protocol import PublishedConfirmationConfig
from .recurrent_published_training_v5 import RecurrentMatchTrainer
from .fast_reproducible_training import fast_numerical_runtime


class PublishedConfirmationTrainer(ConfirmationCheckpointMixin,RecurrentMatchTrainer):
    def __init__(self,env,config,**kwargs):
        if not isinstance(config,PublishedConfirmationConfig):
            raise ValueError('Require an explicit published confirmation/validation config')
        super().__init__(env,config,**kwargs)
        self._bind_confirmation()
        digest=hashlib.sha256(self.implementation_sha256.encode())
        for name in ('published_confirmation_protocol.py','published_confirmation_training.py'):
            digest.update(name.encode());digest.update(Path(__file__).with_name(name).read_bytes())
        self.implementation_sha256=digest.hexdigest()

    def checkpoint_metadata(self):
        if self.runtime!=fast_numerical_runtime():
            raise ValueError('Recurrent producer numerical runtime changed')
        return {**super().checkpoint_metadata(),
            'actor_family':'paper_based_MATCH_recursive_uint8_PPO_port',
            'exact_original_paper_score_reproduced':False,
            'environment_signature':self.environment_signature,
            'prefix_forward':'complete_current_parameters_from_reset',
            'prefix_gradient':self.config.prefix_gradient,
            'actor_forward_batch':self.config.num_envs,
            'minibatch_selection':'after_complete_current_parameter_full_batch_replay',
            'normalizer':{'kind':'fixed_identity','updates':0}}

    def load_checkpoint(self,path):
        state=super().load_checkpoint(path)
        lengths=np.asarray(state.episode_steps)
        if np.any(lengths<0) or np.any(lengths>=self.config.episode_length):
            raise ValueError('Invalid complete recurrent history length')
        return state
