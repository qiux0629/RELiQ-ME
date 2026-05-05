# RELiQ-ME

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
- `rl`：加载 DQN/GNN checkpoint，在候选中心节点中选择动作

## 强化学习中心节点选择

安装基础依赖后，如需训练或使用 RL 中心选择策略，再安装：

```bash
pip install -r standalone_quantum_env/requirements-rl.txt
```

训练一个小规模 DQN checkpoint：

```bash
python standalone_quantum_env/train_multipartite_rl.py \
  --episodes 100 \
  --n-router 30 \
  --multipartite-targets 3 \
  --ghz-simulator auto \
  --ghz-shots 1024 \
  --checkpoint ./output/multipartite_rl.pt
```

使用训练后的策略进行多体请求评估：

```bash
python standalone_quantum_env/create_env.py \
  --n-router 30 \
  --n-data 2 \
  --request-type multipartite \
  --multipartite-targets 3 \
  --center-strategy rl \
  --rl-policy-path ./output/multipartite_rl.pt \
  --ghz-simulator auto \
  --ghz-shots 1024 \
  --ghz-gate-noise 0.01 \
  --ghz-readout-error 0.005 \
  --summary-json ./output/multipartite_rl_summary.json
```

`--ghz-simulator auto` 会优先尝试 QPanda3；未安装 QPanda3 时自动回退到 numpy shots 模拟。QPanda3 后端使用 `NoiseModel + DensityMatrixSimulator`，通过 `--ghz-gate-noise` 给 H/CNOT 加 depolarizing noise，通过 `--ghz-readout-error` 配置读出错误。摘要中会输出 `ghz_simulator_backend`、`rl_policy`、`selected_center_q_value`、`rl_reward` 和噪声模拟细节。

RL 奖励规则：

- 成功生成完整 GHZ 态：奖励为完整 `ghz_fidelity`
- 找到所有 terminal 到中心的路径但 GHZ 未成功：奖励为 `0.5 * ghz_input_fidelity`
- 候选中心不可达：奖励为 `0`
