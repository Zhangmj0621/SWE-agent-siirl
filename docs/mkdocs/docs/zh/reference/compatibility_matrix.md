# 兼容性矩阵

*siirl-agentic 支持的 Python、PyTorch、CUDA 及关键依赖版本信息。*

## Python

| Python 版本 | 状态  |
| --------- | --- |
| 3.10      | 支持  |
| 3.11      | 支持  |
| 3.12      | 支持  |
| < 3.10    | 不支持 |

## PyTorch

| PyTorch 版本 | 状态  | 备注        |
| ---------- | --- | --------- |
| >= 2.8.0   | 支持  | 最低要求版本    |
| 2.7.x      | 未测试 | 可能工作，但不保证 |
| < 2.7      | 不支持 |           |

## CUDA

| CUDA 版本 | 状态   | 备注                 |
| ------- | ---- | ------------------ |
| 12.x    | 支持   | 推荐                 |
| 11.8    | 尽力支持 | 需要兼容的 PyTorch 构建版本 |

## 关键依赖

```yaml title="核心依赖"
ray[default]         >= 2.53.0     # 分布式 Actor 编排
transformers         >= 4.57.0     # 模型加载、分词
sglang               >= 0.5.5      # Rollout 推理引擎
sglang-router        >= 0.3.0      # 多引擎负载均衡
accelerate           >= 1.12.0     # 模型并行工具
datasets             >= 4.4.0      # 数据集加载
wandb                >= 0.23.0     # 实验跟踪
tensorboard          >= 2.20.0     # 训练可视化
loguru               >= 0.7.3      # 结构化日志
fastapi              >= 0.127.0    # AIO 工具服务器 API
aiohttp              >= 3.13.0     # 异步 HTTP 客户端
hydra-core           >= 1.3.2      # 配置管理
omegaconf            (via hydra)   # CLI 点分语法配置解析
```

## 可选 GPU 依赖

```yaml title="GPU 依赖 (pip install siirl-agentic[gpu])"
flash-attn           >= 2.8.2      # FlashAttention-2 加速
triton               >= 3.4.0      # 自定义 kernel 编译
torch-memory-saver   >= 0.0.9      # colocated 模式 GPU 内存优化
```

## 硬件要求

### 最低配置（开发/测试）

-   1× NVIDIA GPU（A100 40GB 或同等配置）
-   64 GB 系统内存
-   100 GB 磁盘空间

### 推荐配置（生产训练）

-   8× NVIDIA A100 80GB 或 H100
-   每节点 512 GB 系统内存
-   高速互联网络（NVLink / InfiniBand）
-   共享文件系统（NFS 或并行文件系统）

### 多节点

-   所有节点必须有相同的 GPU 配置
-   Ray 集群必须预先初始化或自动启动
-   节点间网络延迟 < 1ms（同一数据中心）

## 模型支持

| 模型系列                    | 状态   | 备注                  |
| ----------------------- | ---- | ------------------- |
| Qwen2 / Qwen2.5 / Qwen3 | 支持   | 主要开发目标              |
| LLaMA 3 / 3.1           | 支持   |                     |
| DeepSeek-V2/V3          | 支持   | MoE 架构              |
| 其他 HuggingFace 模型       | 尽力支持 | 需要兼容的 tokenizer 和架构 |

## 部署模式

| 模式            | GPU 要求                    | 说明                       |
| ------------- | ------------------------- | ------------------------ |
| Separated（分离） | actor_gpus + rollout_gpus | 训练和 rollout 使用专用 GPU     |
| Colocated（共置） | 共享 GPU 池                  | Actor 和 rollout 分时共享 GPU |

详见 [部署模式](../guides/deployment_modes.md)。

## 相关文档

-   [安装](../get_started/installation.md) — 分步安装说明
-   [配置参考](config_reference.md) — 完整参数文档
