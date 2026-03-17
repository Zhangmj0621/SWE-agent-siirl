# 安装指南

*完成 siirl-agentic 安装并在本地系统上验证。*

## 环境要求

-   Python >= 3.10
-   CUDA >= 12.4（GPU 支持）
-   PyTorch >= 2.8.0
-   Ray >= 2.53.0

### 已验证环境

```text
组件                 版本
─────────────────────────────
Python               3.10+
PyTorch              2.8.0
CUDA                 12.8
Flash Attention      2.8.2
Transformers         4.57.0
Ray                  2.53.0
SGLang               0.5.5.post3
Triton               3.4.0
```

!!! note
    最低要求的 CUDA 版本为 **12.4**。上表中的已验证版本（12.8）是开发团队验证过的版本。其他 CUDA 12.x 版本应该也能正常工作，但未经官方测试。

!!! warning "Python 版本要求"
    siirl-agentic 需要 **Python >= 3.10**。安装前请先确认版本：
    ```bash
    python --version
    ```
    如果显示 `Python 3.9.x` 或更早版本，请先升级环境再继续。

!!! tip "CUDA 12.x 集群用户"
    如果你在 CUDA 12.x 集群上，请使用 `pip install -e '.[gpu]'` 安装，以获取 flash-attn 和 triton。如果只需 CPU 环境探索，基础安装（`pip install -e '.[dev]'`）无需 GPU 扩展即可运行。

## 从源码安装（推荐）

``` bash
git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic
pip install -e ".[dev]"
```

## GPU 安装（Flash Attention + Triton）

``` bash
git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic
pip install -e ".[gpu]"
```

## 完整安装（所有可选依赖）

``` bash
git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic
pip install -e ".[all]"
```

## 从源码安装（固定版本）

``` bash
git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic

# 先安装 PyTorch（CUDA 12.4+）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 安装 Flash Attention
pip install flash-attn==2.8.2 --no-build-isolation

# 安装本包
pip install -e ".[dev]"
```

## 验证安装

运行以下三个检查，确认安装完整、环境已就绪：

``` bash
# 1. 检查 siirl 包版本
python -c "import siirl; print(siirl.__version__)"

# 2. 检查 CUDA 可用性与 GPU 数量
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 3. 验证 Ray 集群管理
ray start --head && ray status && ray stop
```

**第 1 步预期输出：**

```
0.1.0
```

!!! note
    版本号由 `setuptools_scm` 管理，来源于 git tag。如果你从 git clone 安装，可能会看到类似 `0.1.0.dev123+gabcdef` 而非 `0.1.0` 的版本号。这是正常现象。

**第 2 步预期输出：**

```
CUDA: True, GPUs: 8
```

GPU 数量应与机器上的物理 GPU 数量一致。如果显示 `CUDA: False`，请检查 CUDA 驱动是否已安装，以及是否使用了 `pip install -e '.[gpu]'` 而非基础 dev 安装。

**第 3 步预期输出：**

```
Local node IP: 127.0.0.1
...
Node status
---------------------------------------------------------------
Active:
 1 node(s) with resources: ...
Stopped the local Ray instance.
```

``` bash
# 额外检查：GPU 支持与 CUDA 版本
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, version: {torch.version.cuda}')"
```

**预期输出：**

```
CUDA: True, version: 12.8
```

!!! tip "Python 版本很重要"
    Python 3.10 是最低要求，但推荐使用 Python 3.11 或 3.12 以获得更好性能。如果遇到 `SyntaxError` 或 import 错误，在进一步排查前先用 `python --version` 确认 Python 版本。

!!! tip "仅 CPU 安装"
    如果没有 GPU，可安装不含 GPU 扩展的基础包：`pip install -e ".[dev]"`。这对配置验证和单元测试非常有用。跳过 Flash Attention 步骤——它需要在支持 CUDA 的 GPU 上编译。

## 环境变量

``` bash
# 日志
export LOGURU_LEVEL=INFO
export SIIRL_LOG_DIRECTORY=siirl_logs

# Ray
export RAY_memory_monitor_refresh_ms=0

# 分布式训练
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

## 安装成功的标志

完成本指南后，你应该看到：

- `python -c "import siirl; print(siirl.__version__)"` 打印 `0.1.0`（或形如 `0.1.0.dev123+gabcdef` 的开发版本号）
- `python -c "import torch; print(torch.cuda.is_available())"` 打印 `True`
- 无 import 错误或缺少依赖的警告
- Flash Attention 加载无报错：`python -c "import flash_attn; print(flash_attn.__version__)"`

如果出现问题，请参阅[故障排除](../reference/troubleshooting.md)。

## 下一步

- [快速开始](quickstart.md) — 运行第一个 GRPO 训练任务，端到端验证安装是否正确
- [系统工作原理](../concepts/how_it_works.md) — 深入了解异步训练循环和核心组件，再开始配置
