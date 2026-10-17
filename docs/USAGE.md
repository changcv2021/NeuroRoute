# 环境与使用

## 运行位置

训练、仿真、依赖安装、大数据下载和完整测试应在合适的计算节点完成。仓库保留 Slurm 检查；不要在 HPC 登录节点运行这些工作。公共入口本身不提交作业，也不选择账户、分区或资源。使用所在集群的有效配置申请一张 GPU，并为不同种子分别设置输出目录。

## 固定依赖

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| JAX / JAXLIB / CUDA12 plugin / PJRT | 0.8.2 |
| MuJoCo / mujoco-mjx | 3.6.0 |
| Brax | 0.14.2 |
| Flax | 0.11.2 |
| Optax | 0.2.6 |
| MuJoCo Playground commit | `4057c147714b6ac09b395377f1a1724bbeacc4d3` |
| MuJoCo Menagerie commit | `1b86ece576591213e2b666ebf59508454200ca97` |

完整依赖版本和包哈希位于 `manifests/requirements_jax082.lock`。以下命令在计算节点运行，`NR_WORK` 应指向适合存放环境和缓存的数据盘。使用集群支持的 Python 3.12。

```bash
export NR_CODE="$PWD"
export NR_WORK="/your/persistent/work/NeuroRoute"
mkdir -p "$NR_WORK/software" "$NR_WORK/cache"
python3.12 -m venv "$NR_WORK/env"
export NR_PY="$NR_WORK/env/bin/python"
export PIP_CACHE_DIR="$NR_WORK/cache/pip"
export UV_CACHE_DIR="$NR_WORK/cache/uv"
"$NR_PY" -m pip install 'uv==0.12.13'

git clone --filter=blob:none https://github.com/google-deepmind/mujoco_playground.git \
  "$NR_WORK/software/mujoco_playground"
git -C "$NR_WORK/software/mujoco_playground" checkout --detach \
  4057c147714b6ac09b395377f1a1724bbeacc4d3
mkdir -p "$NR_WORK/software/mujoco_playground/mujoco_playground/external_deps"
git clone --filter=blob:none https://github.com/google-deepmind/mujoco_menagerie.git \
  "$NR_WORK/software/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie"
git -C "$NR_WORK/software/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie" \
  checkout --detach 1b86ece576591213e2b666ebf59508454200ca97

"$NR_WORK/env/bin/uv" pip sync --python "$NR_PY" --require-hashes \
  "$NR_CODE/manifests/requirements_jax082.lock"
"$NR_PY" -m pip install --no-deps --no-build-isolation \
  -e "$NR_WORK/software/mujoco_playground" -e "$NR_CODE"
"$NR_PY" -m pip check
```

运行前设置：

```bash
export PYTHONPATH="$NR_CODE:$NR_CODE/src:$NR_CODE/recovery_v1:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_enable_scatter_determinism_expander=true"
```

公共训练入口也会补全未设置的运行环境变量和仓库导入路径。已有 XLA_FLAGS 若缺少所需确定性选项，入口会报错，请检查后设置。代码校验不代替驱动、CUDA、显存或算子兼容性验证。

## 训练和评估

```bash
"$NR_PY" "$NR_CODE/scripts/train_flycns.py" \
  --budget 0.5 --variant full --seed 101 \
  --output "$NR_WORK/outputs/flycns_b050_seed101" --evaluate-final
```

配置保留默认 PPO 超参数：8192 个并行环境、unroll 长度 20、batch size 256、32 个 minibatches、每批 4 次更新、学习率 0.0003、请求 200M 环境步。训练器按完整更新块运行，实际步数可能向上取整。课程切换位置由代码定义。`configs/flycns_v2.json` 的 `pi_bio` 是模型使用的方向先验参数。

`--output` 必须是尚不存在的目录。程序保存本次配置、进度和策略 checkpoint；`--evaluate-final` 在训练结束后执行固定评估命令。`--save-training-state` 额外保存包含优化器和环境状态的 checkpoint，可能占用较多空间。公共入口当前只支持新训练；恢复接口位于 `recovery_v2/ppo_train.py`，尚未提供公共的自动续跑调度器。

评估输入在 `configs/behavior.json`：每段 250 个控制步，依次执行前进、侧移、转向、前进并反向转向。`recovery_v2/behavior.py` 计算命令跟踪、存活和通信指标。这里的通信成本为逻辑接口传输位数，不是硬件网络流量。

更换模型变体使用 `--variant no_bio`、`no_curriculum`、`no_age`、`no_opt` 或 `bio_v1`；目标预算使用 `--budget 0.75`、`0.5` 或 `0.25`。独立训练任务可以各占一张 GPU 并行运行。

## 连接组处理

下载清单仅包含公开输入数据的版本、地址、大小与校验信息，没有本地提取结果。

```bash
"$NR_PY" scripts/fetch_connectome.py --data-dir "$NR_WORK/data/banc"
"$NR_PY" -m neuroroute.connectome.extract --help
"$NR_PY" -m neuroroute.connectome.derive_prior --help
```

获取数据、校验大文件和提取矩阵都属于计算/传输工作，按所在集群规则执行。

## 验证与发布范围

`python3 tools/check_release.py` 只使用标准库，检查发布文件列表、保留源码哈希、Python/JSON/TOML 语法及不应发布的文件类型。它不导入 JAX 或启动训练。GPU 入口本次只完成静态检查和参数解析检查；尚未重新执行训练或完整测试。

需要验证单元测试时，在配置好的计算节点运行 `python -m pytest tests`。测试源码不依赖本地历史实验结果；依赖安装与运行平台仍需满足对应测试要求。

所有生成输出应保留在仓库之外，或置于 `.gitignore` 排除的目录。发布文件清单采用明确列举，新增文件需要审核后更新 `release/source_snapshot.json` 的 `published_files`。
