# Changelog

*All notable changes to siirl-agentic will be documented here.*

*This project uses [setuptools_scm](https://github.com/pypa/setuptools_scm) for version management.*

## v0.1.0 (Latest)

!!! tip "Versions are derived automatically from git tags via `setuptools_scm`. See `pyproject.toml` for configuration."

### Features

| Category        | Feature                                                     | Details                                                                                                                                                                |
| --------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training**    | Async multi-turn agentic RL                                 | Overlapping rollout and training via `trainer.async_factor`; async state machine with `AgentState` transitions (PENDING -> GENERATING -> PROCESSING_ENV -> TERMINATED) |
| **Algorithms**  | PPO and GRPO with dual-clip                                 | `adv_estimator: ppo` (GAE-based) or `grpo` (group-relative outcome); dual-clip PPO via `clip_ratio_c`; Dr.GRPO variant (`norm_adv_by_std_in_grpo: false`)              |
| **Rollout**     | SGLang-based inference engine                               | Chunked prefill, token-based dynamic batching (`use_dynamic_batch`), router-based load balancing                                                                       |
| **AgentFlow**   | Protocol with dynamic method injection                      | Three-method contract: `preprocess()`, `generate()`, `reward()` on `Sample` objects with typed `AgentMeta` generics                                                    |
| **Tool infra**  | AIO three-layer tool scheduling                             | Proxy -> Scheduler -> Sandbox architecture; `ToolEnv` base class with OpenAI function schema; supports Hermes and GPT-OSS tool formats                                 |
| **SWE**         | SWE-bench agent integration                                 | Docker/K8s/kr8s sandbox environments; `MiniSWEAgent` implementation; SWE-bench and SWE-Factory runtimes                                                                |
| **Checkpoint**  | Megatron distributed checkpointing                          | TP/PP/DP resharding on resume; save contents: `model`, `optimizer`, `extra`, `hf_model`; experimental `async_save`                                                     |
| **Deployment**  | Separated and colocated modes                               | `trainer.colocate: false` (dedicated actor/rollout GPUs) or `true` (shared GPUs with offload); `validate_reuse_train_gpus` for scaled validation                       |
| **Off-policy**  | Off-policy data support                                     | `off_policy_step` controls version staleness tolerance; `off_policy_strategy: fifo` or `oldest_first`                                                                  |
| **Batching**    | Dynamic batching with token scheduling                      | `use_dynamic_batch: true` with `max_tokens_per_gpu`; FLOPs-based workload balancing (`use_workload_balance`)                                                           |
| **Rewards**     | 8 built-in reward categories                                | GSM8K, MATH/AIME, DAPO Math, Prime Math (Numina), Code (SandboxFusion), Geometry3K, MM-Eureka, Search-R1 QA -- covering 28+ data sources                               |
| **Logging**     | WandB metrics integration                                   | `trainer.logger: [console, wandb]`; configurable `log_val_generations`                                                                                                 |
| **Memory**      | CPU offload for parameters, gradients, and optimizer states | `param_offload`, `grad_offload`, `optimizer_offload` on `MegatronArguments`; ref model `param_offload`                                                                 |
| **Parallelism** | Full Megatron parallelism stack                             | TP, PP, DP, context parallel, expert model parallel, expert tensor parallel, sequence parallel, virtual pipeline parallel                                              |

### Known Limitations

| Limitation                                                | Status                                       | Workaround                                                |
| --------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------- |
| Async checkpoint save                                     | Experimental (`checkpoint.async_save: true`) | Use synchronous save for production                       |
| MoE expert routing tracking                               | Basic -- per-token expert lists only         | Manual analysis via `Sample.experts` field                |
| No built-in LoRA/PEFT                                     | Not yet supported                            | Fine-tune full weights or use external adapters pre-merge |
| `siiRL` and `siirl-agentic` share the `siirl` module name | Cannot co-install                            | Use separate virtual environments                         |
