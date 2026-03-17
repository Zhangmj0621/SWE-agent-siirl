# siirl-agentic 文档优化方案

> 参照 ROLL (https://alibaba.github.io/ROLL/docs/Overview/) 文档结构，结合 siirl-agentic 差异化特性制定。

---

## 一、现状评估

### 1.1 现有文件清点

**已完成（英文）**

| 路径 | 状态 | 质量评估 |
|------|------|---------|
| `en/index.rst` | 已完成 | 结构完整，章节合理 |
| `en/highlights/why_agentic_rl.rst` | 已完成 | 内容丰富，对比表格清晰 |
| `en/highlights/pluggable_agentflow_protocol.rst` | 已完成 | 代码示例 + Mermaid 流程图，质量高 |
| `en/user_guide/grpo_training.rst` | 已完成 | 参数表 + 常见问题，覆盖完整 |
| `en/user_guide/ppo_training.rst` | 已完成 | - |
| `en/user_guide/tool_env_and_swe.rst` | 已完成 | - |
| `en/user_guide/deployment_modes.rst` | 已完成 | - |
| `en/user_guide/validate_reuse_and_eval_scaling.rst` | 已完成 | - |
| `en/user_guide/checkpoint_resume.rst` | 已完成 | - |
| `en/user_guide/metrics_and_evaluation.rst` | 已完成 | - |
| `en/advanced/performance_tuning.rst` | 已完成 | - |
| `en/faq/troubleshooting.rst` | 已完成 | 分类详细，覆盖主要场景 |

**仅中文版，英文缺失**

| 路径 | 中文状态 | 优先级 |
|------|---------|-------|
| `highlights/native_agentic_trajectory_training` | 已有 zh | P0 |
| `highlights/mpmd_async_execution_engine` | 已有 zh | P0 |
| `get_started/quickstart` | 已有 zh | P0 |
| `get_started/first_agentic_training_job` | 已有 zh | P0 |
| `user_guide/configuration_system` | 已有 zh | P0 |
| `user_guide/agentic_multiturn` | 已有 zh | P0 |
| `developer_guide/code_structure` | 已有 zh | P1 |

**完全缺失（中英均无）**

| 路径 | 优先级 | 说明 |
|------|-------|------|
| `highlights/aio_elastic_agentic_tool_infrastructure` | P0 | 核心差异化特性 |
| `concepts/design_philosophy` | P1 | 设计思想 |
| `concepts/architecture_overview` | P1 | 架构总览 |
| `concepts/async_training_lifecycle` | P1 | 异步训练生命周期 |
| `get_started/installation` | P0 | 用户第一步 |
| `user_guide/aio_tool_infrastructure` | P0 | AIO 使用指南 |
| `advanced/failure_propagation` | P2 | 故障传播机制 |
| `reference/config_reference` | P1 | 配置参考 |
| `reference/module_map` | P2 | 模块映射 |
| `reference/compatibility_matrix` | P2 | 兼容性矩阵 |
| `developer_guide/adding_new_executor_or_flow` | P1 | 自定义扩展 |
| `developer_guide/contributing` | P1 | 贡献指南 |

---

## 二、ROLL 文档结构对比分析

### ROLL 顶层结构
```
Overview
Getting Started
  ├── Installation
  ├── Quick Start (Single-Node / Multi-Node / DevPod)
  ├── Debugging Guide          ← 独立调试页
  └── FAQ
User Guides
  ├── Configuration            ← 多子页（backend、LoRA、FP8等）
  ├── Pipeline                 ← 每种 Pipeline 独立页（RLVR/DPO/Distill/Agentic）
  ├── Algorithms               ← 每种算法独立页（GRPO/PPO/GSPO等）
  ├── Agentic                  ← 专项（StarPO/GiGPO/Tool Use）
  ├── Advanced Features        ← 异步/Checkpoint/GPU复用/模型转换
  ├── Tracker & Metrics
  └── Hardware Support         ← Ascend 等
Development
  ├── Architecture
  └── Developer Guide
```

### siirl-agentic 当前顶层结构
```
Highlights
Concepts
Get Started
User Guide                     ← 算法和功能混在一起
Advanced
Reference
Developer Guide
FAQ
```

### 主要差距

| 维度 | ROLL | siirl-agentic 现状 | 建议 |
|------|------|-------------------|------|
| 算法文档 | 每个算法独立页，顶层 section | 混入 User Guide | 提升为独立 Algorithms section |
| Pipeline 文档 | 独立 Pipeline section | 无 | 新增 Pipeline section |
| Agentic 专项 | 独立 Agentic section，含论文级内容（StarPO/GiGPO） | 分散在 User Guide | 整合为独立 Agentic section |
| 调试指南 | Getting Started 下独立页 | 与 FAQ 合并 | 拆分为独立 Debugging 页 |
| 硬件支持 | 独立 Hardware Support section | 无 | 新增（至少 CUDA 兼容性） |
| Overview 落地页 | 有专门 Overview 页，含定位说明 | 无，从 Highlights 开始 | 新增 Overview 落地页 |
| 模型格式转换 | 有 checkpoint 转换文档 | checkpoint_resume 部分覆盖 | 补充模型格式转换专页 |
| AIO 工具基础设施 | Tool Use Guide | 无专页（仅在 highlights 提到） | 重点新增，这是核心差异化 |

---

## 三、优化后目录结构

```
docs/
├── en/
│   ├── index.rst                              # 更新 toctree
│   │
│   ├── overview/                              # [NEW] 落地页（对标 ROLL Overview）
│   │   └── index.rst                         # 定位、架构图、快速导航
│   │
│   ├── highlights/                            # 保留，强化差异化
│   │   ├── why_agentic_rl.rst                ✓ 已完成
│   │   ├── native_agentic_trajectory_training.rst  [补英文]
│   │   ├── pluggable_agentflow_protocol.rst  ✓ 已完成
│   │   ├── mpmd_async_execution_engine.rst   [补英文]
│   │   └── aio_elastic_agentic_tool_infrastructure.rst  [NEW]
│   │
│   ├── get_started/                           # [补全]
│   │   ├── installation.rst                  [NEW] Docker/pip/源码三种方式
│   │   ├── quickstart.rst                    [补英文]
│   │   ├── first_agentic_training_job.rst    [补英文]
│   │   └── debugging.rst                     [NEW] 独立调试页（从 FAQ 拆出）
│   │
│   ├── concepts/                              # 保留，内容较好
│   │   ├── design_philosophy.rst             [NEW]
│   │   ├── architecture_overview.rst         [NEW]
│   │   └── async_training_lifecycle.rst      [NEW]
│   │
│   ├── pipelines/                             # [NEW] 对标 ROLL Pipeline section
│   │   ├── grpo_pipeline.rst                 GRPO 端到端流程（数据→rollout→训练）
│   │   ├── ppo_pipeline.rst                  PPO 端到端流程
│   │   └── agentic_pipeline.rst              Agentic 完整流程（含 AIO）
│   │
│   ├── algorithms/                            # [NEW] 算法专区（对标 ROLL Algorithms）
│   │   ├── grpo.rst                          → 从 user_guide 移动/引用
│   │   ├── ppo.rst                           → 从 user_guide 移动/引用
│   │   └── advantage_estimation.rst          [NEW] 优势估计方法对比
│   │
│   ├── agentic/                               # [NEW] Agentic 专区（对标 ROLL Agentic）
│   │   ├── multi_turn_training.rst           [补英文] agentic_multiturn 内容
│   │   ├── agentflow_protocol.rst            → 引用 highlights 内容，扩展使用指南
│   │   ├── tool_env_and_swe.rst              ✓ 已完成（移动或链接）
│   │   └── aio_tool_infrastructure.rst       [NEW] AIO 完整使用指南
│   │
│   ├── user_guide/                            # 保留，聚焦配置和运维
│   │   ├── configuration_system.rst          [补英文]
│   │   ├── deployment_modes.rst              ✓ 已完成
│   │   ├── validate_reuse_and_eval_scaling.rst ✓ 已完成
│   │   ├── checkpoint_resume.rst             ✓ 已完成
│   │   └── metrics_and_evaluation.rst        ✓ 已完成
│   │
│   ├── advanced/                              # 扩充
│   │   ├── performance_tuning.rst            ✓ 已完成
│   │   ├── failure_propagation.rst           [NEW]
│   │   └── model_format_conversion.rst       [NEW] 对标 ROLL 模型转换文档
│   │
│   ├── reference/                             # 参考文档
│   │   ├── config_reference.rst              [NEW] 全量配置项说明
│   │   ├── module_map.rst                    [NEW] 关键模块路径表
│   │   └── compatibility_matrix.rst          [NEW] GPU/Python/依赖版本矩阵
│   │
│   ├── developer_guide/                       # 开发者指南
│   │   ├── code_structure.rst                [补英文]
│   │   ├── adding_new_executor_or_flow.rst   [NEW] 自定义 AgentFlow 教程
│   │   └── contributing.rst                  [NEW] 贡献指南
│   │
│   └── faq/
│       └── troubleshooting.rst               ✓ 已完成（剥离调试部分到 get_started/debugging）
│
└── zh/                                        # 镜像 en/ 结构（中文版补全）
    └── ... (同上结构，优先补全缺失文件)
```

---

## 四、核心内容优化建议

### 4.1 Overview 落地页（最高优先级）

参照 ROLL Overview 页，新增 `en/overview/index.rst`，内容包括：

1. **一句话定位**：siirl-agentic 是什么，与 verl/ROLL/OpenRLHF 的区别
2. **核心差异化特性表格**（4大亮点 vs 竞品）
3. **架构图**（已在 README 中，迁移过来）
4. **快速导航卡片**：新用户路径 / 研究者路径 / 工程师路径
5. **论文/技术报告引用**

### 4.2 新增 Pipelines Section（P0）

ROLL 将每种训练 Pipeline 作为独立文档页，siirl-agentic 应效仿：

**`pipelines/agentic_pipeline.rst`** — 最重要，展示完整 agentic 训练流程：
```
数据准备 (Parquet)
  → DataCoordinator 分发
  → RolloutManager (SGLang + AgentFlow + ToolEnv/AIO)
  → DataBuffer 缓冲
  → TrainerGroup (PPO/GRPO)
  → 参数同步回 RolloutManager
```
每个阶段配置项、常见调优点、监控指标。

### 4.3 AIO 工具基础设施文档（P0，核心差异化）

AIO 是 siirl-agentic 相比 ROLL 最大的差异化，必须有完整文档：

**`agentic/aio_tool_infrastructure.rst`** 应覆盖：
- AIO 三层架构（Proxy → ResourcePool → WorkerManager）
- Holt-Winters 自动扩缩原理
- 部署方式（本地 / 分布式）
- 与 rollout 的集成配置
- 监控和容量规划

### 4.4 调试指南独立（P1）

ROLL 将调试作为 Getting Started 下的独立页（`Debugging Guide`），而非 FAQ。
建议将 `faq/troubleshooting.rst` 中的系统性调试内容迁移到 `get_started/debugging.rst`，保留 FAQ 作为具体问题快速查找。

**`get_started/debugging.rst`** 内容：
- 日志系统（loguru 配置，`LOGURU_LEVEL=DEBUG`）
- Ray Dashboard 使用
- 组件状态检查命令
- 典型调试流程（从症状到定位）

**`faq/troubleshooting.rst`** 保留：
- 按错误类型快速索引的问题列表

### 4.5 算法文档结构化（P1）

将算法从 `user_guide/` 提升到独立 `algorithms/` section：

| 当前位置 | 新位置 | 变更 |
|---------|-------|------|
| `user_guide/grpo_training.rst` | `algorithms/grpo.rst` | 移动，user_guide 保留链接 |
| `user_guide/ppo_training.rst` | `algorithms/ppo.rst` | 移动，user_guide 保留链接 |
| — | `algorithms/advantage_estimation.rst` | 新增，对比 GRPO/GAE/PPO advantage |

### 4.6 Concepts 三页内容建议（P1）

**`concepts/architecture_overview.rst`**：
- 扩展 README 中的架构图
- 每个组件详细说明（DataCoordinator/RolloutManager/TrainerGroup/DataBuffer）
- 数据流动路径
- 与 verl/ROLL 架构对比

**`concepts/async_training_lifecycle.rst`**：
- 异步训练完整时序图
- on-policy vs off-policy 数据流
- `async_factor` 和 `off_policy_step` 的工作原理
- DataBuffer 容量与 staleness 权衡

**`concepts/design_philosophy.rst`**：
- 为什么选 Ray（vs Torchrun/DeepSpeed ZeRO）
- 为什么选 SGLang（vs vLLM）
- MPMD vs SPMD 的 trade-off
- 框架设计原则

### 4.7 Installation 页面（P0）

`get_started/installation.rst` 应覆盖三种安装方式：

```rst
.. tab-set::

   .. tab-item:: Docker (推荐)
      docker pull sii-research/siirl-agentic:latest

   .. tab-item:: pip
      pip install -e ".[gpu]"

   .. tab-item:: 源码
      git clone ... && pip install -e ".[dev]"
```

加上：CUDA/Python 版本要求、依赖冲突说明（siirl vs siiRL 的模块名冲突）。

---

## 五、index.rst 更新方案

### 英文 index.rst 新结构

```rst
.. toctree::
   :caption: Overview
   overview/index

.. toctree::
   :caption: Highlights
   highlights/why_agentic_rl
   highlights/native_agentic_trajectory_training
   highlights/pluggable_agentflow_protocol
   highlights/mpmd_async_execution_engine
   highlights/aio_elastic_agentic_tool_infrastructure

.. toctree::
   :caption: Get Started
   get_started/installation
   get_started/quickstart
   get_started/first_agentic_training_job
   get_started/debugging

.. toctree::
   :caption: Concepts
   concepts/design_philosophy
   concepts/architecture_overview
   concepts/async_training_lifecycle

.. toctree::
   :caption: Pipelines              ← 新增
   pipelines/agentic_pipeline
   pipelines/grpo_pipeline
   pipelines/ppo_pipeline

.. toctree::
   :caption: Algorithms              ← 从 User Guide 提升
   algorithms/grpo
   algorithms/ppo
   algorithms/advantage_estimation

.. toctree::
   :caption: Agentic                 ← 新增，核心差异化专区
   agentic/multi_turn_training
   agentic/agentflow_protocol
   agentic/tool_env_and_swe
   agentic/aio_tool_infrastructure

.. toctree::
   :caption: User Guide              ← 保留，聚焦配置和运维
   user_guide/configuration_system
   user_guide/deployment_modes
   user_guide/validate_reuse_and_eval_scaling
   user_guide/checkpoint_resume
   user_guide/metrics_and_evaluation

.. toctree::
   :caption: Advanced
   advanced/performance_tuning
   advanced/failure_propagation
   advanced/model_format_conversion

.. toctree::
   :caption: Reference
   reference/config_reference
   reference/module_map
   reference/compatibility_matrix

.. toctree::
   :caption: Developer Guide
   developer_guide/code_structure
   developer_guide/adding_new_executor_or_flow
   developer_guide/contributing

.. toctree::
   :caption: FAQ
   faq/troubleshooting
```

---

## 六、实施计划

### Phase 1 — 补全缺失文件（P0，最高优先级）

解决英文版空白，让文档可以完整构建：

| 文件 | 来源 | 工作量 |
|------|------|-------|
| `en/get_started/installation.rst` | 从 INSTALL.md + README 整合 | 小 |
| `en/get_started/quickstart.rst` | 翻译 zh 版本 | 小 |
| `en/get_started/first_agentic_training_job.rst` | 翻译 zh 版本 | 小 |
| `en/highlights/native_agentic_trajectory_training.rst` | 翻译 zh 版本 | 小 |
| `en/highlights/mpmd_async_execution_engine.rst` | 翻译 zh 版本 | 小 |
| `en/highlights/aio_elastic_agentic_tool_infrastructure.rst` | 从代码分析 + README | 中 |
| `en/user_guide/configuration_system.rst` | 翻译 zh 版本 | 中 |
| `en/user_guide/agentic_multiturn.rst` | 翻译 zh 版本 | 中 |
| `en/agentic/aio_tool_infrastructure.rst` | 从 AIO/ 代码分析 | 大 |

### Phase 2 — 结构重组（P1）

1. 新增 `pipelines/`, `algorithms/`, `agentic/` 三个 section
2. 将 `user_guide/grpo_training.rst` 和 `ppo_training.rst` 移到 `algorithms/`
3. 将 `user_guide/tool_env_and_swe.rst` 移到 `agentic/`
4. 更新 en/zh 两个 `index.rst`

### Phase 3 — 新增核心内容（P1-P2）

| 文件 | 说明 |
|------|------|
| `overview/index.rst` | 落地页，含竞品对比表格 |
| `concepts/` 三页 | 架构、设计哲学、异步生命周期 |
| `pipelines/agentic_pipeline.rst` | 端到端流程图 |
| `get_started/debugging.rst` | 独立调试指南 |
| `algorithms/advantage_estimation.rst` | 算法对比 |
| `advanced/model_format_conversion.rst` | Megatron ↔ HF 格式转换 |
| `developer_guide/adding_new_executor_or_flow.rst` | 自定义 AgentFlow 教程 |
| `reference/config_reference.rst` | 全量配置参数说明 |

### Phase 4 — 中文版同步

1. 将 Phase 1-3 新增的英文内容同步翻译为中文
2. 补全 zh/index.rst 与 en/index.rst 结构对齐

---

## 七、内容质量标准

参照现有高质量文件（`why_agentic_rl.rst`、`pluggable_agentflow_protocol.rst`）的写作规范：

1. **每页开头**：`Who this is for` + `What you will get`（两行 admonition）
2. **代码示例**：所有配置块使用 `.. code:: yaml`，所有代码使用 `.. code:: python`
3. **对比表格**：关键功能差异用 RST 表格，而非列表
4. **流程图**：复杂流程用 `.. mermaid::` 块
5. **代码引用**：在文档中标注关键实现文件（`file:line` 格式）
6. **Common Issues**：每个功能页末尾附常见问题表（symptom → cause → fix）

---

## 八、工作量估算

| Phase | 文件数 | 预估工作量 | 说明 |
|-------|-------|-----------|------|
| Phase 1（补全缺失） | 9 | 中等 | 大部分是翻译或整合现有内容 |
| Phase 2（结构重组） | 2（index） | 小 | 更新 toctree + 移动文件 |
| Phase 3（新增内容） | 10 | 较大 | 需要深入分析代码 |
| Phase 4（中文同步） | 15+ | 中等 | 翻译为主 |
| **合计** | **36+** | | |

---

## 九、与 ROLL 的最终差异定位

优化后的 siirl-agentic 文档应在以下维度明确区分于 ROLL：

| 维度 | ROLL | siirl-agentic |
|------|------|--------------|
| 训练规模 | 通用，含大规模分布式 | 专注 Agentic 场景 |
| 推理引擎 | vLLM + SGLang | SGLang 原生 |
| 算法广度 | 8+ 算法（TOPR/GSPO/RAFT++等） | PPO + GRPO，专精 Agentic |
| 核心差异化 | 异步并行 Rollout | **AIO 弹性工具基础设施** + AgentFlow 协议 |
| 工具集成 | Tool Use Guide | **AIO 三层分布式工具调度（核心优势）** |
| 硬件支持 | Ascend + GPU | GPU 优先 |

文档主轴应始终围绕：**"为什么 agentic RL 需要专门的框架，以及 siirl-agentic 如何解决这个问题"**。
