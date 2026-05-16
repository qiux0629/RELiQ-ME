# Standalone Quantum Environment

这个目录是从 RELiQ 中抽出来的“环境创建最小集合”，用于在新项目里单独复用量子网络环境生成与环境实例化逻辑。

## 当前主线

当前项目主线是“在 RELiQ-style 二体 routing backbone 上扩展多体 GHZ 分发任务”，不是和 RELiQ 原二体 routing 任务比较谁更强。RELiQ-style routing 在这里负责多条 center-to-terminal 分发路径；本文重点是多体请求建模、GHZ center selection、中心节点本地 state-carving GHZ 生成，以及把预生成 GHZ 粒子分发到用户终端。

默认多体模式是论文主线：

```text
multipartite request
  -> RELiQ-style GNN/heuristic center selection
  -> local state-carving GHZ generation at center
  -> RELiQ-style path/routing-policy distribution to terminals
  -> user-side multipartite GHZ success/failure
```

旧的“terminal-to-center 二体纠缠全部完成后再 GHZ fusion”仍通过 `--ghz-establishment-mode fusion_after_bipartite` 保留为对照消融，不再作为默认主线。

推荐实验也应围绕多体 GHZ 指标展开：

- GNN center policy 与启发式 center strategy 的对比。
- `path_estimator`、`routing_policy`、`dynamic_physical` 三类 backend 的消融。
- local state-carving GHZ 成功概率与 fidelity 对 success/fidelity/time/hops/reward 的影响。
- 默认物理 profile 与 `state_carving_2025_high_fidelity` 高保真上界假设的对比。

`train_bipartite_routing.py` 和 `evaluate_bipartite_routing.py` 主要用于验证底层二体 backbone 是否可用，不作为最终论文主贡献。

## 目录结构

- `create_env.py`: 独立运行入口，直接创建环境并输出摘要
- `visualization.py`: 环境可视化工具，绘制拓扑、链路状态和请求位置
- `env/`: 环境与网络核心代码
- `data/all_fidelities.npy`: 量子链路保真度查表数据
- `util.py`: 环境代码依赖的最小工具函数
- `requirements.txt`: 运行这个目录所需的最小依赖

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

从 `standalone_quantum_env` 目录运行：

```bash
python create_env.py \
  --n-router 100 \
  --n-data 1 \
  --summary-json ./output/summary.json \
  --save-plot ./output/topology.png \
  --no-node-labels
```

也可以从项目根目录运行：

```bash
python standalone_quantum_env/create_env.py \
  --n-router 100 \
  --n-data 1 \
  --summary-json ./output/summary.json \
  --save-plot ./output/topology.png \
  --no-node-labels
```

注意：`create_env.py` 内部会切换工作目录到 `standalone_quantum_env`。如果使用相对路径，输出文件会保存到 `standalone_quantum_env/output/` 下。

## 设置生成节点数量

节点数量由 `--n-router` 参数控制。例如生成 50 个节点：

```bash
python create_env.py \
  --n-router 50 \
  --n-data 1 \
  --save-plot ./output/topology_50.png
```

常用参数：

- `--n-router`: 量子网络节点数量，默认 100
- `--n-data`: 路由请求/agent 数量，默认 1
- `--save-plot`: 保存可视化图片路径
- `--summary-json`: 保存环境摘要 JSON 路径
- `--no-node-labels`: 节点较多时隐藏节点编号
- `--show-plot`: 直接弹出图窗

可视化图包含：

- 节点颜色：节点 swap probability
- 边颜色：边上活跃量子链路的平均 fidelity
- 边宽度：边上活跃量子链路数量
- 红色方块：请求起点
- 绿色三角：请求目标点
- 橙色空心圆：当前 packet 位置

如果节点数量较多，可以隐藏节点编号：

```bash
python create_env.py \
  --n-router 100 \
  --n-data 4 \
  --save-plot ./output/topology.png \
  --no-node-labels
```

也可以直接弹出图窗：

```bash
python create_env.py --n-router 12 --n-data 2 --show-plot
```

## 作为代码复用

你可以在新项目里直接导入：

```python
from create_env import create_quantum_env, build_parser
from visualization import visualize_environment
```

或者直接使用环境类：

```python
from env.entanglementenv import EntanglementEnv
from env.quantum_network import QuantumNetwork
```

## 说明

- 这个目录不依赖 `src/main.py`
- 已经包含环境创建所需的代码与 fidelity 数据
- 当前系统包含 GHZ center policy、RELiQ-style 二体 routing policy、可切换二体 backend 和 GHZ fusion sampling

## 一对多多体纠缠请求

`create_env.py` 可以生成一对多请求并按“多个 target 同时路由 -> 中心等待 -> GHZ fusion -> 成功/失败”的流程输出结果：

```bash
python create_env.py \
  --n-router 30 \
  --n-data 2 \
  --request-type multipartite \
  --multipartite-targets 3 \
  --center-strategy balanced \
  --ghz-fusion-gate-fidelity 0.98 \
  --ghz-fusion-success-probability 0.9 \
  --reward-mode time_fidelity \
  --reward-latency-weight 0.01 \
  --reward-resource-weight 0.02
```

相关实现位于 `env/multipartite.py`，包含：

- 请求生成：`source -> targets`
- 中心节点策略：`balanced`、`median`、`minimax`、`min-latency`、`max-fidelity`、`random`
- 时延拆分：采样、打包、经典传播、GNN 推理、Agent 决策、BSM、存储、GHZ fusion
- 并发路由：source 和所有 target 到中心的路径同时建立，中心等待最慢路径
- Fidelity：路径使用 RELiQ 的逐跳 noisy swap，GHZ fusion 再按 gate fidelity 引入 depolarizing noise
- 成功判定：所有路径完成、GHZ fidelity 达到阈值，且 fusion 成功概率非零
- GNN 特征：节点层与边层特征名、特征矩阵形状和 edge index 结构

`EntanglementEnv.step` 也新增了可选 `time_fidelity` 奖励模式。默认仍为 `legacy`，不会改变原有实验行为。

## 标准 GNN 编码器

`env/gnn_encoder.py` 提供 `QuantumGraphGNNEncoder`，用于把 `build_gnn_feature_package(...)` 生成的数据编码为节点 embedding 和整图 embedding：

```python
from env.gnn_encoder import GraphBatch, QuantumGraphGNNEncoder
from env.multipartite import build_gnn_feature_package

features = build_gnn_feature_package(env.network, terminals=terminals, center=center)
batch = GraphBatch.from_feature_package(features)
encoder = QuantumGraphGNNEncoder.from_feature_package(features)
encoded = encoder(batch.node_features, batch.edge_index, batch.edge_features)
```

编码器是边特征感知的多层消息传递 GNN。当前项目用同一个编码器支持多体 GHZ center selection 和 RELiQ-style 二体 next-hop routing policy。

## RELiQ-style 二体 Backbone

`train_bipartite_routing.py` 训练一个二体 GNN routing policy，用作多体 GHZ 流程的底层二体 backbone。它不是本文主贡献，主要用于让 `terminal -> center` 二体纠缠建立接近 RELiQ-style learned routing。模型输入整张量子网络图、当前节点、目标节点、已访问节点和链路状态，输出 `0=idle` 或 `1..neighbor_count=当前节点邻边槽位`。

训练默认配置是 30 节点、TTL 12、3000 条 imitation samples 和 500 个 RL episodes。教师策略可选择 `shortest_path`、`max_fidelity` 或 `balanced`，也可以用 `--curriculum-n-router 30,50,100` 做课程训练。未到达 target 的 rollout 会把 fidelity 记为 0 并施加 `--unreached-penalty`；`--allow-wait-action` 只在当前节点没有合法下一跳时允许等待。

```bash
python train_bipartite_routing.py \
  --routing-teacher balanced \
  --curriculum-n-router 30,50,100 \
  --unreached-penalty 2.0 \
  --checkpoint ./output/experiments/bipartite_routing_30.pt \
  --metrics-json ./output/experiments/bipartite_routing_30_train.json
```

评估并和 baseline 对比，仅作为 backbone sanity check：

```bash
python evaluate_bipartite_routing.py \
  --checkpoint ./output/experiments/bipartite_routing_30.pt \
  --policy all \
  --n-router 30 \
  --eval-episodes 100 \
  --metrics-json ./output/experiments/bipartite_routing_30_eval.json
```

## 多体 GHZ 成功概率评估

```bash
python evaluate_multipartite_center.py \
  --policy balanced \
  --n-router 1000 \
  --multipartite-targets 2 \
  --ghz-establishment-mode prebuilt_ghz_distribution \
  --state-carving-fidelity 0.9999 \
  --state-carving-local-gate-fidelity 0.9999 \
  --eval-episodes 20 \
  --metrics-json ./output/multipartite_1000_prebuilt_ghz.json
```

统一评估脚本默认执行“中心先生成 GHZ，再分发到终端”的论文主线，并输出 GHZ 成功率、本地 GHZ 成功/失败率、保真度、总时延和总跳数。早期单独的 `evaluate_multipartite_ghz.py` 入口已经合并到 `evaluate_multipartite_center.py`。

## 多体中心节点选择训练

上层模型 `MultipartiteCenterPolicy` 负责为 `source -> targets` 任务选择 GHZ generation/distribution center。默认 `prebuilt_ghz_distribution` 模式下，中心节点先用 state-carving 生成本地 GHZ，然后底层 RELiQ-style backend 负责 center-to-terminal 分发路径、保真度、时延与资源消耗。旧的 `fusion_after_bipartite` 模式才会把这些路径解释为 terminal-to-center 二体纠缠和中心 GHZ fusion。

```bash
python train_multipartite_center.py \
  --n-router 50 \
  --multipartite-targets 2 \
  --imitation-episodes 100 \
  --train-episodes 200 \
  --reward-latency-weight 0.01 \
  --reward-resource-weight 0.02 \
  --checkpoint ./output/multipartite_center_policy.pt
```

```bash
python evaluate_multipartite_center.py \
  --checkpoint ./output/multipartite_center_policy.pt \
  --policy all \
  --ghz-establishment-mode prebuilt_ghz_distribution \
  --n-router 100 \
  --multipartite-targets 2 \
  --eval-episodes 20
```

接入二体 routing policy：

```bash
python evaluate_multipartite_center.py \
  --checkpoint ./output/multipartite_center_policy.pt \
  --policy all \
  --ghz-establishment-mode prebuilt_ghz_distribution \
  --bipartite-backend routing_policy \
  --routing-checkpoint ./output/experiments/bipartite_routing_30.pt \
  --n-router 100 \
  --multipartite-targets 2 \
  --eval-episodes 20
```

## 动态链路级物理后端

`dynamic_physical` 后端用于更真实地评估多体 GHZ 建立过程。默认 `prebuilt_ghz_distribution` 模式下，它会先在中心本地生成 GHZ，再逐 hop 消耗真实 `edge.links` 把 GHZ 粒子分发到终端；失败原因主要来自动态链路生成、swap 或分发 fidelity 不达标。只有在 `--ghz-establishment-mode fusion_after_bipartite` 时，才会走旧的 photon-to-atom 写入、memory decay、readout / atom-to-photon conversion 和最终 GHZ fusion 流程。

```bash
python evaluate_multipartite_center.py \
  --policy balanced \
  --bipartite-backend dynamic_physical \
  --ghz-establishment-mode prebuilt_ghz_distribution \
  --n-router 30 \
  --multipartite-targets 2 \
  --eval-episodes 10 \
  --metrics-json ./output/dynamic_physical_n30.json
```

默认物理 profile 为 `single_atom_cavity_2015`：

- photon-to-atom 写入成功率 `0.39`、写入 fidelity `0.86`
- atom-to-photon readout 成功率 `0.69`、readout fidelity `0.88`
- memory lifetime `0.22s`

这些参数来自相关量子存储文献，不是 RELiQ 原始参数：写入/readout 参考 [Heralded Storage of a Photonic Quantum Bit in a Single Atom](https://pubmed.ncbi.nlm.nih.gov/26196608/)，memory lifetime 参考 [An efficient quantum light-matter interface with sub-second lifetime](https://www.nature.com/articles/nphoton.2016.51)。

本地 GHZ 生成使用独立的 state-carving 参数，默认 profile 为 `state_carving_2025_high_fidelity`，其中 `state_carving_fidelity=0.9999`、`state_carving_local_gate_fidelity=0.9999`。如果使用 sequential 构造，局部 GHZ fidelity 按 `F_ESC^(N-1) * F_gate^(N-2)` 估计，局部成功概率按 `p_ESC^(N-1)` 估计。

物理 memory profile 也可以使用 `state_carving_2025_high_fidelity` 作为高保真上界假设。该 profile 只把 photon-to-atom/atomic-entanglement interface fidelity 设为 `0.9999`，其他成功概率、readout fidelity 和 memory lifetime 仍沿用默认 profile，避免把“高条件保真度”误写成“所有物理步骤都确定成功”。如果需要理想成功概率上界，应额外显式覆盖 success probability 参数。

这篇 state-carving 论文对本项目的帮助主要有三点：

- 支持把光-物质接口的条件态保真度作为高保真上界设置，例如 `state_carving_2025_high_fidelity`。
- 提醒我们把 fidelity 和 success probability 分开建模；coupling loss、switch loss、detection loss 主要影响成功概率。
- 提供多体 graph/cluster state 的物理动机，可用于讨论 GHZ/graph-state 多体纠缠扩展，但不等价于本项目已经复现完整 state-carving 协议。

可用以下参数覆盖默认值：

```bash
--state-carving-profile state_carving_2025_high_fidelity
--state-carving-success-probability 1.0
--state-carving-fidelity 0.9999
--state-carving-local-gate-fidelity 0.9999
--state-carving-construction sequential

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

动态后端默认额外输出 `memory_wait_time`、`memory_decay_loss`、`write_success`、`readout_success`、`conversion_failure_rate` 和 `memory_failure_rate`。在 `prebuilt_ghz_distribution` 下，这些 memory/conversion 字段通常为 0 或 `null`，因为论文主线不再把每条 terminal-to-center 二体纠缠先写入原子 memory 后再 fusion。逐事件调试字段 `physical_timeline` 和 `stored_entanglements` 体积较大，默认不写入 metrics JSON；需要排查物理链路细节时再加 `--include-physical-details`。

`max-photonic-route-attempts` 用于 photonic center-to-terminal 分发路径的整条重建；当设置大于 1 时，动态后端会在 routing policy 路径和 shortest-simple 候选路径之间重试，并选择成功候选中 fidelity 最高的一条。`max-write-attempts` 和 `max-readout-attempts` 只作用于旧的 `fusion_after_bipartite` 模式；`memory-slots-per-node=0` 表示当前抽象下不限制中心节点 memory slot；`purification-rounds` 是可选的简化 purification 模型，用于观察 fidelity/资源 trade-off。

1000 节点 routing policy smoke eval 可用 `--routing-cache-static-encoding` 加速。该参数每条 center-to-terminal route 只编码一次整图，适合大图可行性测试，不建议作为最终严格论文结果。

## 推荐多体 GHZ 实验

建议用已有 `output/experiments/center_30.pt` 或 `output/experiments/center_ac_30.pt`，以及 `output/experiments/bipartite_routing_fixed_curriculum.pt` 跑 30/50/100 节点对照实验。实验重点是多体 GHZ 成功率、保真度、时延和资源消耗，不是二体 routing 单项性能。

- `path_estimator`：规则路径估计器。
- `routing_policy`：训练好的二体 GNN next-hop policy。
- `dynamic_physical`：链路级动态物理执行，包含光子-原子写入、memory decay、readout 和 atom-to-photon conversion。

主实验建议报告：

- GNN center policy vs `balanced` / `median` / `max-fidelity` / `min-latency` / `random`
- `path_estimator` vs `routing_policy` vs `dynamic_physical`
- `prebuilt_ghz_distribution` 主线 vs `fusion_after_bipartite` 对照消融
- local state-carving GHZ success/fidelity sensitivity
- `single_atom_cavity_2015` vs `state_carving_2025_high_fidelity`
- 30/50/100 节点训练与评估，500/1000 节点作为泛化 smoke/eval

```bash
python evaluate_multipartite_center.py \
  --checkpoint output/experiments/center_30.pt \
  --policy all \
  --bipartite-backend path_estimator \
  --n-router 30 \
  --multipartite-targets 2 \
  --eval-episodes 30 \
  --metrics-json output/experiments/model_effect_path_n30.json
```

```bash
python evaluate_multipartite_center.py \
  --checkpoint output/experiments/center_30.pt \
  --policy all \
  --bipartite-backend routing_policy \
  --routing-checkpoint output/experiments/bipartite_routing_30.pt \
  --n-router 30 \
  --multipartite-targets 2 \
  --eval-episodes 30 \
  --metrics-json output/experiments/model_effect_routing_n30.json
```

```bash
python evaluate_multipartite_center.py \
  --checkpoint output/experiments/center_30.pt \
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
  --metrics-json output/experiments/model_effect_dynamic_perfect_n30.json
```

汇总所有正式 JSON：

```bash
python summarize_experiments.py \
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
