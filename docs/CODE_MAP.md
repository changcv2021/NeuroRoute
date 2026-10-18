# Code guide

The package contains the final FlyCNS-v2 implementation and the components needed to train and evaluate it.

| Module | Responsibility |
| --- | --- |
| `environment` | Go1 environment, observation ownership, joint mapping, and physical fault adapter |
| `models.flycns` | Cached actor, conditional action distribution, recurrent replay, and inference |
| `models.memory` | Receiver message buffers and packet ages |
| `models.routing` | Causal gate probabilities and conditional motor distributions |
| `models.common` | Shared neural layers, logical packet transport, and reference critic |
| `models.probability` | Stable KL diagnostics and the training network factory |
| `training.ppo` | PPO, communication budget dual updates, and state callbacks |
| `training.acting` | Rollouts carrying receiver communication state |
| `training.curriculum` | Budget schedules for 75%, 50%, and 25% |
| `training.loss` | PPO loss, budget penalty, and biological prior term |
| `training.diagnostics` | Auxiliary PPO diagnostics |
| `evaluation.behavior` | Fixed command evaluation and communication accounting |
| `evaluation.metrics` | Tracking, survival, and per-case summaries |
| `evaluation.rendering` | Video and tracking-plot generation from recorded trajectories |
| `checkpoints` | Policy serialization and integrity checks |
| `connectome` | Anatomical input selection, normalization, prior extraction, and sensitivity analysis |

All modules are under the installed `neuroroute` package.

```text
scripts/train_flycns.py
  -> environment.FlyCNSGo1
  -> models.probability.make_networks
  -> training.ppo.train
       -> training.acting
       -> training.curriculum
       -> training.loss
```

The default `full` configuration enables the biological prior, budget curriculum, message age features, adaptive dual optimization, and receiver memory. The four component ablations use the same model implementation and retain receiver memory.

Communication accounting measures logical transmitted bits. Each sent packet contains 32 float32 values and four overhead bits. Missing packets cost zero and retain the previous receiver message. Packet ages are updated as part of the receiver state.
