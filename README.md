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
