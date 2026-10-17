# 代码导航

## 当前 FlyCNS-v2

| 模块 | 职责 |
| --- | --- |
| `src/neuroroute/environment.py` | Go1 环境适配、腿部映射与扰动机制 |
| `src/neuroroute/central_v1.py` | 中央策略基础、观测契约和策略 checkpoint |
| `src/neuroroute/full_v1.py` | 局部/全局网络及逻辑消息传输计数 |
| `src/neuroroute/sparse_v1.py` | 因果 Bernoulli 路由、条件电机动作与 PERIODIC 调度 |
| `src/neuroroute/sparse_fast_v1.py` | 稀疏策略计算实现 |
| `src/neuroroute/flycns_v2.py` | 接收端消息缓存、消息年龄、路由上下文及策略网络 |
| `recovery_v2/stable_probability.py` | 训练所使用的稳定 KL 诊断网络工厂 |
| `recovery_v2/acting.py` | 携带通信状态的 rollout |
| `recovery_v2/ppo_train.py` | PPO、通信预算对偶更新及训练状态回调 |
| `recovery_v2/loss.py` | 状态重放、PPO 损失、预算项和生物先验项 |
| `recovery_v2/extension_20260914/curriculum.py` | 为训练器安装覆盖三个预算的课程函数 |
| `recovery_v1/observability.py` | PPO 辅助诊断 |
| `recovery_v2/behavior.py` | 命令跟踪评估及逻辑传输量计算 |
| `recovery_v1/behavior.py` | 轨迹汇总与渲染辅助函数 |

公共入口的调用链：

```text
scripts/train_flycns.py
  ├─ configs/flycns_v2.json
  ├─ FlyCNSGo1 + stable_probability.make_networks
  ├─ extension_20260914.curriculum.install
  └─ ppo_train.train
       ├─ acting
       └─ loss.routing_loss
```

目录名称保留原实现的导入路径。`recovery_v1/` 和 `recovery_v2/` 在此是代码模块，并不包含历史实验记录。

## 连接组及其他实现

`src/neuroroute/connectome/` 包含数据筛选、归一化、方向先验构造、控制先验和敏感性分析程序。数据版本与下载地址由 `manifests/connectome_manifest.json` 提供，筛选规则位于 `configs/connectome_banc_v888.json`。

`src/neuroroute/` 下其余模块保存早期策略、基线、评估和统计计算代码。名称中含 `results` 的 Python 模块是计算程序，不包含已有实验结果。部分早期接口要求调用者提供自己的配置或输入文件；公共运行示例面向当前 FlyCNS-v2。

`release/source_snapshot.json` 记录原样保留文件的 SHA256。公共入口、文档及从原协议提取的纯输入配置单独维护，不伪装成原有实验代码。
