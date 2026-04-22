# Standalone Quantum Environment

这个目录是从 RELiQ 中抽出来的“环境创建最小集合”，用于在新项目里单独复用量子网络环境生成与环境实例化逻辑。

## 目录结构

- `create_env.py`: 独立运行入口，直接创建环境并输出摘要
- `env/`: 环境与网络核心代码
- `data/all_fidelities.npy`: 量子链路保真度查表数据
- `util.py`: 环境代码依赖的最小工具函数
- `requirements.txt`: 运行这个目录所需的最小依赖

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python create_env.py \
  --n-router 12 \
  --n-data 2 \
  --summary-json ./output/summary.json \
  --save-plot ./output/topology.png
```

## 作为代码复用

你可以在新项目里直接导入：

```python
from create_env import create_quantum_env, build_parser
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
