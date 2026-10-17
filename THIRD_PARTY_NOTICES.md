# Third party notices

Original upstream notices remain in redistributed source files. No new repository-wide license is assigned to original NeuroRoute code.

| Component | Use | License notice |
| --- | --- | --- |
| Brax | Derived PPO and acting implementation in `recovery_v2/` and other training modules | [Apache License 2.0](LICENSES/Brax-Apache-2.0.txt) |
| MuJoCo Playground | Go1 environment dependency and adapted interfaces | [Apache License 2.0](LICENSES/MuJoCo-Playground-Apache-2.0.txt) |
| OpenAI multi-agent particle environments | Code in `third_party/openai_mpe/` | [MIT](third_party/openai_mpe/LICENSE.txt) |

Upstream repositories: [Brax](https://github.com/google/brax), [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), and [OpenAI multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs). Pinned dependency versions and upstream commits are documented in `docs/USAGE.md`.

BANC input data locations and checksums are in `manifests/connectome_manifest.json`. Raw datasets and derived data artifacts are not redistributed. Dataset access and terms remain those of the source release.
