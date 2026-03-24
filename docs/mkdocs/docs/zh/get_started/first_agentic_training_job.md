# 首个 Agentic 训练任务

*启动一个完整的 GRPO 训练任务，其中 agent 在多轮 rollout 中使用搜索工具。*

## 前置条件

-   siirl-agentic 已安装并验证（参见 [安装指南](installation.md)）
-   已完成 [快速开始](quickstart.md)（基础 GRPO 训练可正常运行）
-   **AIO 工具基础设施** 已克隆并安装。AIO 是 monorepo 中的独立仓库：

    ``` bash
    # 从 monorepo 根目录（siirl-agentic/ 的上级目录）
    cd AIO
    pip install -e .
    ```

    如果你没有 AIO 仓库，请联系团队获取访问权限或参阅 [AIO 工具基础设施](../guides/aio_tool_infrastructure.md)。

## 什么是"Agentic"训练？

在标准 RL 训练中，模型生成单个响应并获得奖励。在 **agentic** 训练中：

1.  模型生成的文本可能包含**工具调用**（如搜索查询、代码执行）
2.  工具调用通过 AIO 基础设施在**真实环境中执行**
3.  工具响应作为环境观测**追加到对话中**
4.  模型基于工具响应**继续生成**
5.  这个多轮循环重复直到终止（达到最大轮数或模型主动结束）
6.  **只有模型生成的 token** 参与策略梯度（环境 token 被屏蔽）

!!! tip "工具环境启动顺序"
    务必在启动训练脚本**之前**先启动 AIO 基础设施。Rollout 引擎在启动时会尝试连接 AIO Proxy——如果 Proxy 未运行，rollout 初始化将失败并报错 `AIOSearchTool: Error getting server`。

## 第一步：配置多轮 Rollout

在训练配置中添加多轮配置：

``` yaml
rollout:
  flow_function: naive
  multiturn:
    env_type: tool_env
    max_env_turns: 5            # 最大工具交互轮数（默认：1）
    max_assistant_turns: 10     # 最大模型生成轮数（默认：1）
    max_parallel_calls: 4       # 每个样本的并发工具调用数（默认：1）
    max_env_response_length: 256  # 环境响应的最大字符数，用于截断
    env_response_truncate_side: middle
    env_path: /path/to/tool_env_config.yaml
    env_kwargs:
      tool_format: hermes
```

!!! warning
    `max_env_response_length` 基于**字符数**截断，而非 token 数。值为 256 表示 256 个字符，根据分词器的不同，这可能少于或多于 256 个 token。

## 第二步：配置工具环境

创建工具环境配置文件（`tool_env_config.yaml`）：

``` yaml
tools:
  - name: search
    type: aio_search
    config:
      topk: 3
```

## 第三步：启动 AIO 基础设施

训练开始前，先启动 AIO Proxy 和 WorkerManager：

``` bash
# 启动 AIO Proxy
python -m aio.Scheduler.proxy --config aio_config.yaml

# 在每个工具节点启动 WorkerManager
python -m aio.Scheduler.Resources.worker_manager --proxy-url http://proxy-host:8080
```

## 第四步：启动 Agentic 训练

``` bash
# 使用 AIO 示例脚本
cd siirl-agentic
bash examples/AIO/run_qwen3_8b.sh
```

**预期输出：**

```
INFO  | Ray is initialized. Time cost: 150.23 ms
INFO  | MainRunner started. Beginning workflow setup...
INFO  | Initializing DataCoordinator...
SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
INFO  | Starting async training loop...
INFO  | [NaiveFlow] PENDING -> GENERATING (sample 0)
INFO  | [NaiveFlow] GENERATING -> PROCESSING_ENV (sample 0, 2 tool calls)
INFO  | [AIOSearchTool] search query dispatched to worker
INFO  | [NaiveFlow] PROCESSING_ENV -> GENERATING (sample 0, env_turn 1)
INFO  | [NaiveFlow] GENERATING -> TERMINATED (sample 0, reward=1.0)
```

## 第五步：监控训练

关键指标：

-   **reward/mean** — 每步平均奖励（应逐步上升）
-   **rollout/generation_duration** — LLM 生成耗时
-   **rollout/reward_duration** — 奖励计算耗时
-   **rollout/env_turns_mean** — 每个样本的平均工具交互轮数

## 理解轨迹结构

一个典型的 agentic 轨迹如下：

```
[用户]    东京的人口是多少？
[助手]    我来搜索一下这个信息。
          <tool_call>search(query_list=["东京人口"])</tool_call>
[工具]    东京人口约为 1396 万……
[助手]    根据搜索结果，东京人口约为 1396 万人。
```

在 `response_mask` 中（作用于**响应部分**，而非 prompt）：

- 助手轮次 1 的 token → `response_mask = 1`（参与训练，包含在策略梯度中）
- 工具响应的 token → `response_mask = 0`（从 loss 中屏蔽）
- 助手轮次 2 的 token → `response_mask = 1`（参与训练，包含在策略梯度中）

prompt token 在 `prompts` 字段中单独记录，不会参与 loss 计算。

## 最小可运行示例

``` python
# 验证多轮配置是否正确解析
from siirl.params import parse_config, SiiRLArguments

# parse_config() 使用 argparse + OmegaConf.from_cli() 解析 CLI 参数
config = parse_config()
print(f"env_type: {config.rollout.multiturn.env_type}")
print(f"max_env_turns: {config.rollout.multiturn.max_env_turns}")
print(f"max_assistant_turns: {config.rollout.multiturn.max_assistant_turns}")
```

## 常见配置错误

| 错误                             | 结果                  | 正确配置                                     |
| ------------------------------ | ------------------- | ---------------------------------------- |
| `max_env_turns=1`              | Agent 只能进行一次工具调用就终止 | 多步任务使用 `max_env_turns=5`                 |
| 忘记设置 `env_type=tool_env`       | 仅单轮 rollout，工具从不被调用 | 设置 `rollout.multiturn.env_type=tool_env` |
| 首次运行使用 `rollout.n=32`          | OOM 或 rollout 极慢    | 先用 `rollout.n=4`，验证后再扩大                  |
| `max_response_length=512`（默认值） | Agentic 轨迹在第一轮后被截断  | 设置 `data.max_response_length=4096` 或更高   |
| 在 AIO Proxy 启动前运行训练            | Rollout 初始化立即失败     | 先启动 AIO Proxy，再启动训练                      |

## 常见问题

| 现象                                    | 原因                           | 解决方案                               |
| ------------------------------------- | ---------------------------- | ---------------------------------- |
| `AIOSearchTool: Error getting server` | AIO Proxy 未启动                | 运行 `python -m aio.Scheduler.proxy` |
| `TERMINATED after 1 turn`             | `max_assistant_turns=1`      | 增大 `max_assistant_turns`           |
| 工具响应被截断                               | `max_env_response_length` 过小 | 增大该值                               |
| 所有奖励为 0                               | 奖励函数不兼容多轮轨迹                  | 检查自定义奖励函数逻辑                        |

## 多轮 Agentic 训练的成功标志

完成本指南后，你应该看到：

- 日志中出现状态机转换：`PENDING -> GENERATING -> PROCESSING_ENV -> GENERATING -> TERMINATED`
- `rollout/env_turns_mean` 指标**大于 1** — 这确认 agent 正在真实使用工具并生成多轮交互，而不是在第一轮就终止
- `reward/mean` 从非零值开始，并随训练步数上升
- 工具调用成功率可在日志中通过 AIO Proxy 看到：`[AIOSearchTool] search query dispatched to worker`
- AIO Proxy 日志显示收到 `tool_call` 请求并分派给 worker
- 稳定运行后不再出现 `TERMINATED after 1 turn` 警告

**健康 agentic 训练运行的日志示例：**

```
INFO  | [Step 1] rollout done. reward/mean=0.08, env_turns/mean=2.3, tool_calls=184
INFO  | [Step 1] training done. loss=1.51
INFO  | [Step 10] rollout done. reward/mean=0.24, env_turns/mean=2.7, tool_calls=216
INFO  | [Step 10] training done. loss=1.27
```

`env_turns/mean=2.3` 表示 agent 平均每个样本进行 2.3 次工具交互。如果该值停留在 `1.0`，说明 agent 在第一轮后就终止了——请检查配置中的 `max_env_turns` 和 `max_assistant_turns`。

如果出现问题，请参阅[故障排除](../reference/troubleshooting.md)。

## 下一步

- [Agentic 多轮训练](../guides/agentic_multiturn.md) — 深入了解多轮配置、状态机行为与 loss masking 机制
- [AIO 工具基础设施](../guides/aio_tool_infrastructure.md) — 为生产环境配置并扩展分布式工具调度系统
- [系统工作原理](../concepts/how_it_works.md) — 建立对刚才运行的异步训练循环的系统性认知
