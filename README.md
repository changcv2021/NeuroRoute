# NeuroRoute / FlyCNS

面向 Go1 四足机器人的受限通信控制代码，使用 JAX、MuJoCo MJX 和 Brax PPO。包含 FlyCNS-v2、ASYM、PERIODIC 等策略实现，以及连接组先验提取、通信成本计算和行为评估程序。

本仓库只发布代码、必要配置、依赖和使用说明。实验结果、报告、图表、视频、权重和运行记录保留在本地。

## 代码结构

| 路径 | 用途 |
| --- | --- |
| `src/neuroroute/` | 环境、策略、消息传输、通信指标及连接组处理 |
| `recovery_v2/` | FlyCNS-v2 的 PPO、消息状态重放、损失和行为评估 |
| `recovery_v1/` | 当前训练与评估仍使用的基础辅助函数 |
| `configs/` | 网络、训练、预算消融和评估输入配置 |
| `scripts/train_flycns.py` | 从新目录开始训练的公共入口 |
| `scripts/fetch_connectome.py` | 获取固定版本的连接组输入数据 |
| `tests/` | 通信、策略、环境和连接组的单元测试源码 |
| `manifests/` | 依赖锁文件及外部数据下载清单 |
| `release/source_snapshot.json` | 保留源码的 SHA256 清单 |

完整模块说明见 [CODE_MAP](docs/CODE_MAP.md)，环境准备与使用方式见 [USAGE](docs/USAGE.md)。

## 开始使用

Python 3.12。在安装科学计算依赖前，可以先执行轻量静态检查和查看配置：

```bash
python3 tools/check_release.py
python3 scripts/train_flycns.py --help
python3 scripts/train_flycns.py --budget 0.25 --variant full --seed 101 --print-config
```

安装依赖后，在合适的 Slurm GPU allocation 中启动一个独立单卡任务：

```bash
python3 scripts/train_flycns.py --budget 0.25 --variant full --seed 101 \
  --output /your/work/outputs/flycns_b025_seed101 --evaluate-final
```

支持 `0.75`、`0.5`、`0.25` 三个目标预算；可用变体为 `full`、`no_bio`、`no_curriculum`、`no_age`、`no_opt` 和 `bio_v1`。通信预算是训练目标，实际逻辑传输量由发送记录计算。

公共入口复用保留的模型和训练代码，从随机初始化开始，不依赖本地历史实验的准入文件。它是新增的使用入口；本次整理完成的是源码完整性与静态检查，尚未对这个入口重新运行 GPU 训练。更早的实现保留为库模块，历史调度器及其运行记录不在发布范围内。

第三方组件见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。项目原创代码尚未指定仓库级许可证。
