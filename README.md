# RELiQ-ME

## 当前主线

本项目主线不是和 RELiQ 做二体 routing 性能竞争，而是在 RELiQ-style routing backbone 上扩展多体 GHZ 分发任务。RELiQ 在这里承担底层路径选择与链路评估 backbone 的角色；本文贡献集中在多体请求建模、GHZ center selection、中心节点本地 state-carving GHZ 生成，以及把预生成 GHZ 粒子分发到用户终端。

默认执行流程已经调整为论文主线：

```text
multipartite request
  -> RELiQ-style GNN/heuristic center selection
  -> local state-carving GHZ generation at center
  -> RELiQ-style path/routing-policy distribution to terminals
  -> user-side multipartite GHZ success/failure
```

旧的“先建立多条 terminal-to-center 二体纠缠，再在中心 GHZ fusion”的流程仍保留为 `--ghz-establishment-mode fusion_after_bipartite` 对照消融。

推荐论文叙事：

- **基础层**：沿用 RELiQ-style quantum network、二体路径/链路 fidelity、noisy swap 与资源约束。
- **扩展层**：把一个 `source -> targets` 多体请求转化为中心生成 GHZ 后的多条 `center -> terminal` 分发路径。
- **决策层**：训练 GNN center policy 选择 GHZ generation/distribution center，而不是把二体 routing policy 作为主贡献。
- **物理层**：默认模拟 state-carving 本地 GHZ 生成与分发路径损耗；旧 fusion 模式可选模拟 photon-to-atom storage、memory decay、readout、atom-to-photon conversion 和 GHZ fusion sampling。
- **消融层**：比较不同 center strategy、不同 backend、不同物理 profile 对 GHZ success/fidelity/time/hops/reward 的影响。

因此，`train_bipartite_routing.py` 和 `evaluate_bipartite_routing.py` 主要用于构造和验证 RELiQ-style 二体 backbone；正式结果应优先报告多体 GHZ 指标，而不是声称超过 RELiQ 原二体任务。

## 量子网络环境生成与可视化

项目当前的独立环境入口在 `standalone_quantum_env/create_env.py`，可以生成量子网络环境、输出环境摘要，并保存拓扑可视化图片。

## 安装依赖

```bash
cd standalone_quantum_env
pip install -r requirements.txt
```

## 运行示例

从项目根目录运行：

```bash
python standalone_quantum_env/create_env.py \
  --n-router 100 \
  --n-data 1 \
  --summary-json ./output/summary.json \
  --save-plot ./output/topology.png \
  --no-node-labels
```

生成内容：

- `summary.json`：环境摘要，包括节点数、边数、请求数量等
- `topology.png`：量子网络拓扑可视化图片

注意：脚本内部会切换到 `standalone_quantum_env` 目录。如果使用相对路径，图片会生成到 `standalone_quantum_env/output/topology.png`。想固定保存到项目根目录，可以使用绝对路径。

## 设置节点数量

节点数量由 `--n-router` 控制：

```bash
python standalone_quantum_env/create_env.py \
  --n-router 50 \
  --n-data 1 \
  --save-plot ./output/topology_50.png
```

常用参数：

- `--n-router 100`：生成 100 个量子网络节点，默认值也是 100
- `--n-data 1`：生成 1 个路由请求/agent
- `--save-plot PATH`：保存环境可视化图片
- `--summary-json PATH`：保存环境摘要 JSON
- `--no-node-labels`：节点较多时隐藏节点编号，避免图片过乱
- `--show-plot`：直接弹出图窗，不一定保存文件

## 可视化含义

- 节点颜色：节点的 swap probability
- 边颜色：边上活跃量子链路的平均 fidelity
- 边宽度：边上活跃量子链路数量，越粗表示链路越多
- 红色方块：请求起点
- 绿色三角：请求目标点
- 橙色空心圆：当前 packet 位置

## 多体纠缠规划与奖励模型

当前代码支持两种多体相关模式：

- `--request-type bipartite`：保留原 RELiQ 二体请求，再把二体端点集合拿去做多体规划评估
- `--request-type multipartite`：生成真正的一对多请求 `source -> targets`，多个 target 同时路由，中心节点等待所有路径完成后执行 GHZ fusion

一对多多体请求示例：

```bash
python standalone_quantum_env/create_env.py \
  --n-router 30 \
  --n-data 2 \
  --request-type multipartite \
  --multipartite-targets 3 \
  --center-strategy balanced \
  --ghz-fusion-gate-fidelity 0.98 \
  --ghz-fusion-success-probability 0.9 \
  --reward-mode time_fidelity \
  --reward-latency-weight 0.01 \
  --reward-resource-weight 0.02 \
  --summary-json ./output/multipartite_summary.json
```

新增输出字段：

- `multipartite_execution`：一对多请求、中心节点、并发路由、中心等待、GHZ fusion、成功/失败结果
- `multipartite_plan`：兼容旧规划模式，选择出的中心节点、各终端到中心的路径、估计时延、保真度、成功概率和奖励
- `multipartite_delay_model`：采样、打包、经典传播、GNN 推理、BSM、存储和 GHZ fusion 的时延参数
- `multipartite_fidelity_model`：RELiQ-style fidelity 阈值、GHZ fusion gate fidelity 和 fusion 成功概率
- `multipartite_reward_weights`：成功、路径可达、保真度、时延和资源消耗对应的奖励权重
- `gnn_feature_package`：节点特征、边特征和 edge index 的形状及特征名，可直接作为 GNN 输入结构参考

## 标准 GNN 数据编码

`standalone_quantum_env/env/gnn_encoder.py` 提供了标准 PyTorch GNN 编码器 `QuantumGraphGNNEncoder`，可以直接消费 `build_gnn_feature_package(...)` 生成的整图节点特征、双向 `edge_index` 和边特征：

```python
from env.gnn_encoder import GraphBatch, QuantumGraphGNNEncoder
from env.multipartite import build_gnn_feature_package

features = build_gnn_feature_package(env.network, terminals=[source, *targets], center=center)
batch = GraphBatch.from_feature_package(features)
encoder = QuantumGraphGNNEncoder.from_feature_package(
    features,
    hidden_dim=128,
    output_dim=128,
    num_layers=3,
    dropout=0.1,
)
encoded = encoder(batch.node_features, batch.edge_index, batch.edge_features)

node_embeddings = encoded.node_embeddings
graph_embedding = encoded.graph_embedding
```

这个编码器使用边特征参与每一层消息传递，并返回节点级 embedding 与整图 embedding。当前项目同时用它训练多体 GHZ center selection 和 RELiQ-style 二体 next-hop routing policy。

## RELiQ-style 二体 Backbone

`train_bipartite_routing.py` 会训练一个二体 GNN routing policy，用作多体 GHZ 流程的底层二体 backbone。它不是本文主贡献，主要作用是替代规则路径估计器，让 `terminal -> center` 二体纠缠建立更接近 RELiQ-style learned routing。输入包括整张量子网络图、当前节点、目标节点、已访问节点和链路状态；输出动作为：

- `0`：idle
- `1..neighbor_count`：选择当前节点的第几个邻边槽位作为下一跳

训练分两段，也支持 30/50/100 课程训练：

- imitation learning：模仿 `shortest_path`、`max_fidelity` 或 `balanced` 下一跳，先得到可用路由策略
- RL fine-tuning：用成功率、fidelity、latency、hop/resource reward 微调；未到达 target 的 rollout 会把 fidelity 记为 0 并施加 `--unreached-penalty`
- `--allow-wait-action` 只在当前节点没有合法下一跳时允许等待，避免模型把 idle 学成提前停止动作

```bash
python standalone_quantum_env/train_bipartite_routing.py \
  --routing-teacher balanced \
  --curriculum-n-router 30,50,100 \
  --unreached-penalty 2.0 \
  --checkpoint ./output/experiments/bipartite_routing_30.pt \
  --metrics-json ./output/experiments/bipartite_routing_30_train.json
```

评估二体策略并和 baseline 对比，仅作为 backbone sanity check：

```bash
python standalone_quantum_env/evaluate_bipartite_routing.py \
  --checkpoint ./output/experiments/bipartite_routing_30.pt \
  --policy all \
  --n-router 30 \
  --eval-episodes 100 \
  --metrics-json ./output/experiments/bipartite_routing_30_eval.json
```

## 多体 GHZ 成功概率评估

`evaluate_multipartite_center.py` 会对一对多请求执行中心选择、terminal-to-center 路径估计和 GHZ fusion 随机采样。`--ghz-fusion-success-probability` 不再只是理论值，而会作为 Bernoulli 概率决定 fusion 是否成功。早期单独的 `evaluate_multipartite_ghz.py` 入口已经合并进该统一评估脚本。

```bash
python standalone_quantum_env/evaluate_multipartite_center.py \
  --policy balanced \
  --n-router 1000 \
  --multipartite-targets 2 \
  --ghz-fusion-success-probability 0.9 \
  --eval-episodes 20 \
  --metrics-json ./output/multipartite_1000_ghz.json
```

输出包含 `success_rate`、`route_completion_rate`、`fusion_failure_rate`、`mean_ghz_fidelity`、`mean_total_time` 和 `mean_total_hops`。

## 多体中心节点选择模型

当前多体改进目标是：对一个 `source -> targets` 的 GHZ 任务，先由上层 GNN/RL 模型选择 fusion center，再把任务拆成多条 `terminal -> center` 二体纠缠建立过程。底层二体后端可选择：

- `path_estimator`：内置规则路径估计器，默认值
- `routing_policy`：训练好的 RELiQ-style 二体 GNN routing policy

最后中心节点等待所有二体纠缠完成并执行 GHZ fusion。

训练中心选择策略：

```bash
python standalone_quantum_env/train_multipartite_center.py \
  --n-router 50 \
  --multipartite-targets 2 \
  --imitation-episodes 100 \
  --train-episodes 200 \
  --reward-latency-weight 0.01 \
  --reward-resource-weight 0.02 \
  --checkpoint ./output/multipartite_center_policy.pt
```

评估 GNN center policy 并和启发式 center 策略对比：

```bash
python standalone_quantum_env/evaluate_multipartite_center.py \
  --checkpoint ./output/multipartite_center_policy.pt \
  --policy all \
  --n-router 100 \
  --multipartite-targets 2 \
  --eval-episodes 20
```

使用训练好的二体 routing policy 作为多体底层后端：

```bash
python standalone_quantum_env/evaluate_multipartite_center.py \
  --checkpoint ./output/multipartite_center_policy.pt \
  --policy all \
  --bipartite-backend routing_policy \
  --routing-checkpoint ./output/experiments/bipartite_routing_30.pt \
  --n-router 100 \
  --multipartite-targets 2 \
  --eval-episodes 20
```

## 动态链路级物理后端

`dynamic_physical` 后端会逐跳消耗真实 `edge.links`，执行 noisy swap，然后把成功建立的光子二体纠缠写入原子存储。中心节点等待所有 terminal-to-center 原子纠缠完成后，再执行 memory decay、readout / atom-to-photon conversion，最后进入 GHZ fusion。

```bash
python standalone_quantum_env/evaluate_multipartite_center.py \
  --policy balanced \
  --bipartite-backend dynamic_physical \
  --n-router 30 \
  --multipartite-targets 2 \
  --eval-episodes 10 \
  --metrics-json ./output/dynamic_physical_n30.json
```

默认物理 profile 为 `single_atom_cavity_2015`：

- photon-to-atom 写入成功率 `0.39`、写入 fidelity `0.86`
- atom-to-photon readout 成功率 `0.69`、readout fidelity `0.88`
- memory lifetime `0.22s`

参数来源使用相关量子存储文献，不是 RELiQ 原始参数：写入/readout 参考 [Heralded Storage of a Photonic Quantum Bit in a Single Atom](https://pubmed.ncbi.nlm.nih.gov/26196608/)，memory lifetime 参考 [An efficient quantum light-matter interface with sub-second lifetime](https://www.nature.com/articles/nphoton.2016.51)。

也可以使用 `state_carving_2025_high_fidelity` 作为高保真上界假设。该 profile 受 efficient single-photon state-carving 方案启发，只把 photon-to-atom/atomic-entanglement interface fidelity 设为 `0.9999`，其他成功概率、readout fidelity 和 memory lifetime 仍沿用默认 profile，避免把“高条件保真度”误写成“所有物理步骤都确定成功”。如果需要理想成功概率上界，应额外显式覆盖 success probability 参数。

这篇 state-carving 论文对本项目的帮助主要有三点：

- 它支持把光-物质接口的条件态保真度作为高保真上界设置，例如 `state_carving_2025_high_fidelity`。
- 它说明“高 fidelity”和“高 success probability”应分开建模；coupling loss、switch loss、detection loss 主要影响成功概率，不应被混进 fidelity。
- 它提供了多体 graph/cluster state 的物理动机，可用于讨论 GHZ/graph-state 多体纠缠扩展，但不等价于我们已经复现了完整 state-carving 协议。

可用参数覆盖：

```bash
--physical-profile single_atom_cavity_2015
# or: --physical-profile state_carving_2025_high_fidelity
--memory-lifetime-seconds 0.22
--photon-to-atom-success-probability 0.39
--photon-to-atom-fidelity 0.86
--atom-to-photon-success-probability 0.69
--atom-to-photon-fidelity 0.88
--max-write-attempts 3
--max-readout-attempts 3
--max-photonic-route-attempts 3
--memory-slots-per-node 8
--purification-rounds 1
```

动态后端默认额外输出精简诊断指标：

- `memory_wait_time`
- `memory_decay_loss`
- `write_success`
- `readout_success`
- `conversion_failure_rate`
- `memory_failure_rate`

逐事件调试字段 `physical_timeline` 和 `stored_entanglements` 体积较大，默认不写入 metrics JSON；需要排查物理链路细节时再加 `--include-physical-details`。

`max-photonic-route-attempts` 用于 photonic terminal-to-center route 的整条重建；当设置大于 1 时，动态后端会在 routing policy 路径和 shortest-simple 候选路径之间重试，并选择成功候选中 photonic fidelity 最高的一条写入 memory。`max-write-attempts` 和 `max-readout-attempts` 用于 heralded write/readout 失败后的重试；`memory-slots-per-node=0` 表示当前抽象下不限制中心节点 memory slot；`purification-rounds` 是简化 purification 模型，用于观察 fidelity 与资源/时延之间的 trade-off。

1000 节点 smoke eval 可加速 routing policy 后端：

```bash
python standalone_quantum_env/evaluate_multipartite_center.py \
  --checkpoint output/experiments/center_ac_30.pt \
  --policy gnn \
  --bipartite-backend routing_policy \
  --routing-checkpoint output/experiments/bipartite_routing_fixed_curriculum.pt \
  --routing-cache-static-encoding \
  --n-router 1000 \
  --multipartite-targets 2 \
  --eval-episodes 2
```

`--routing-cache-static-encoding` 是近似加速：每条 terminal-to-center route 只编码一次整图，适合大图 smoke/generalization，不建议作为最终严格论文结果。

## 推荐多体 GHZ 实验

推荐用已有 `center_30.pt` 或 `center_ac_30.pt`，以及 `bipartite_routing_fixed_curriculum.pt` 做 30/50/100 节点对照实验。实验重点是多体 GHZ 成功率、保真度、时延和资源消耗，不是二体 routing 单项性能。三类二体后端含义如下：

- `path_estimator`：规则路径估计器，适合看中心选择模型在理想路径估计下的表现。
- `routing_policy`：训练好的二体 GNN next-hop policy，适合看接入二体策略后的表现。
- `dynamic_physical`：链路级动态物理执行，包含真实 `edge.links` 消耗、光子-原子写入、memory decay、readout 和 atom-to-photon conversion。

主实验建议报告：

- GNN center policy vs `balanced` / `median` / `max-fidelity` / `min-latency` / `random`
- `path_estimator` vs `routing_policy` vs `dynamic_physical`
- deterministic GHZ fusion vs stochastic GHZ fusion
- `single_atom_cavity_2015` vs `state_carving_2025_high_fidelity`
- 30/50/100 节点训练与评估，500/1000 节点作为泛化 smoke/eval

示例命令：

```bash
python standalone_quantum_env/evaluate_multipartite_center.py \
  --checkpoint standalone_quantum_env/output/experiments/center_30.pt \
  --policy all \
  --bipartite-backend path_estimator \
  --n-router 30 \
  --multipartite-targets 2 \
  --eval-episodes 30 \
  --metrics-json standalone_quantum_env/output/experiments/model_effect_path_n30.json
```

```bash
python standalone_quantum_env/evaluate_multipartite_center.py \
  --checkpoint standalone_quantum_env/output/experiments/center_30.pt \
  --policy all \
  --bipartite-backend routing_policy \
  --routing-checkpoint standalone_quantum_env/output/experiments/bipartite_routing_30.pt \
  --n-router 30 \
  --multipartite-targets 2 \
  --eval-episodes 30 \
  --metrics-json standalone_quantum_env/output/experiments/model_effect_routing_n30.json
```

```bash
python standalone_quantum_env/evaluate_multipartite_center.py \
  --checkpoint standalone_quantum_env/output/experiments/center_30.pt \
  --policy all \
  --bipartite-backend dynamic_physical \
  --photon-to-atom-success-probability 1.0 \
  --photon-to-atom-fidelity 1.0 \
  --atom-to-photon-success-probability 1.0 \
  --atom-to-photon-fidelity 1.0 \
  --memory-lifetime-seconds 1000 \
  --n-router 30 \
  --multipartite-targets 2 \
  --eval-episodes 30 \
  --metrics-json standalone_quantum_env/output/experiments/model_effect_dynamic_perfect_n30.json
```

汇总正式实验 JSON：

```bash
python standalone_quantum_env/summarize_experiments.py \
  --input-glob "output/experiments/model_effect_*.json" \
  --csv output/experiments/summary.csv \
  --markdown output/experiments/summary.md
```

## 指标解释

- `success_rate`：最终 GHZ 用户态建立成功比例。
- `route_completion_rate`：所有 terminal-to-center 二体链路完成比例。
- `fusion_failure_rate`：路径和 fidelity 达标后，GHZ fusion 随机失败比例。
- `mean_ghz_fidelity`：最终 GHZ 态平均保真度；失败 episode 通常记为 0。
- `mean_total_time`：从任务开始到 GHZ fusion 完成的平均总时间。
- `mean_total_hops`：所有 terminal-to-center 路径的总跳数，反映资源消耗。
- `mean_reward`：当前 reward 函数下的综合得分，包含成功、fidelity、latency、hop/resource。
- `conversion_failure_rate`：`dynamic_physical` 下 photon-to-atom write 或 atom-to-photon readout 失败比例。
- `memory_failure_rate`：memory decay 或 readout 后 fidelity 低于阈值导致失败的比例。
- `failure_reason_counts`：失败原因统计，用来判断瓶颈是 routing、fusion、conversion 还是 memory。

建议训练规模按 `30/50 -> 100 -> 500/1000 eval` 分阶段扩大，不建议直接从 1000 节点训练开始。

Fidelity 计算逻辑：

- 单条路径使用 RELiQ 的逐跳 noisy swap：每经过一个中间节点，调用 `QuantumLink.get_swap_fidelity(f1, f2, gate_error)` 更新端到端 fidelity
- 多体 GHZ 输入 fidelity 为所有 terminal-to-center 路径 fidelity 的乘积
- GHZ fusion 后按 depolarizing gate 近似更新 fidelity，`--ghz-fusion-gate-fidelity` 越低，最终 `ghz_fidelity` 越低
- 成功概率会乘上 `--ghz-fusion-success-probability`

中心节点策略可选：

- `balanced`：综合奖励最大
- `median`：总跳数最小
- `minimax`：最长终端路径最短
- `min-latency`：估计总时延最小
- `max-fidelity`：估计 GHZ 保真度最大
- `random`：随机中心节点
