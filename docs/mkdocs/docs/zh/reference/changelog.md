# 更新日志

*siirl-agentic 的所有重要变更均记录于此。*

*本项目使用 [setuptools_scm](https://github.com/pypa/setuptools_scm) 进行版本管理。*

## v0.1.0（最新版）

!!! tip "版本号通过 `setuptools_scm` 从 git 标签自动派生。详见 `pyproject.toml` 配置。"

### 功能特性

| 类别            | 特性                        | 详情                                                                                                                           |
| ------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **训练**        | 异步多轮 Agentic RL           | 通过 `trainer.async_factor` 实现 rollout 与训练重叠；异步状态机支持 `AgentState` 状态转移（PENDING -> GENERATING -> PROCESSING_ENV -> TERMINATED）  |
| **算法**        | PPO 和 GRPO 及 dual-clip 支持 | `adv_estimator: ppo`（基于 GAE）或 `grpo`（组内相对优势）；dual-clip PPO 通过 `clip_ratio_c` 控制；Dr.GRPO 变体（`norm_adv_by_std_in_grpo: false`） |
| **Rollout**   | 基于 SGLang 的推理引擎           | Chunked prefill、基于 token 的动态 batching（`use_dynamic_batch`）、路由负载均衡                                                            |
| **AgentFlow** | 带动态方法注入的协议                | 三方法契约：`preprocess()`、`generate()`、`reward()`，作用于带类型化 `AgentMeta` 泛型的 `Sample` 对象                                             |
| **工具基础设施**    | AIO 三层工具调度                | Proxy -> Scheduler -> Sandbox 架构；`ToolEnv` 基类配合 OpenAI 函数 schema；支持 Hermes 和 GPT-OSS 工具格式                                    |
| **SWE**       | SWE-bench Agent 集成        | Docker/K8s/kr8s 沙箱环境；`MiniSWEAgent` 实现；SWE-bench 和 SWE-Factory 运行时                                                           |
| **检查点**       | Megatron 分布式检查点           | 恢复时支持 TP/PP/DP 重分片；保存内容：`model`、`optimizer`、`extra`、`hf_model`；实验性 `async_save`                                              |
| **部署**        | 分离模式和共置模式                 | `trainer.colocate: false`（独立的 actor/rollout GPU）或 `true`（共享 GPU 配合 offload）；`validate_reuse_train_gpus` 用于扩展验证               |
| **离策略**       | 离策略数据支持                   | `off_policy_step` 控制版本陈旧容忍度；`off_policy_strategy: fifo` 或 `oldest_first`                                                     |
| **Batching**  | 基于 token 调度的动态 batching   | `use_dynamic_batch: true` 配合 `max_tokens_per_gpu`；基于 FLOPs 的负载均衡（`use_workload_balance`）                                     |
| **奖励**        | 8 大内置奖励类别                 | GSM8K、MATH/AIME、DAPO Math、Prime Math（Numina）、代码（SandboxFusion）、Geometry3K、MM-Eureka、Search-R1 QA -- 覆盖 28+ 数据源               |
| **日志**        | WandB 指标集成                | `trainer.logger: [console, wandb]`；可配置 `log_val_generations`                                                                 |
| **内存**        | 参数、梯度、优化器状态的 CPU offload  | `MegatronArguments` 上的 `param_offload`、`grad_offload`、`optimizer_offload`；ref 模型 `param_offload`                             |
| **并行**        | 完整的 Megatron 并行栈          | TP、PP、DP、上下文并行、专家模型并行、专家张量并行、序列并行、虚拟流水线并行                                                                                    |

### 已知限制

| 限制                                       | 状态                                 | 解决方法                       |
| ---------------------------------------- | ---------------------------------- | -------------------------- |
| 异步检查点保存                                  | 实验性（`checkpoint.async_save: true`） | 生产环境使用同步保存                 |
| MoE 专家路由追踪                               | 基础 -- 仅支持每 token 的专家列表             | 通过 `Sample.experts` 字段手动分析 |
| 无内置 LoRA/PEFT 支持                         | 尚未支持                               | 全量微调或使用外部适配器预合并            |
| `siiRL` 和 `siirl-agentic` 共享 `siirl` 模块名 | 无法同时安装                             | 使用独立的虚拟环境                  |
