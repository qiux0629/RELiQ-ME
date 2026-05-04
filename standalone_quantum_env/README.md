# Standalone Quantum Environment

这个目录是从 RELiQ 中抽出来的“环境创建最小集合”，用于在新项目里单独复用量子网络环境生成与环境实例化逻辑。

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
- 训练模型、策略网络、回放缓存等内容没有放进来

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
