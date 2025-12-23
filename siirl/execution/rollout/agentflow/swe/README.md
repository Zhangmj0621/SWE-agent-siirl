# SWE 任务

SWE 任务通常有 multi-turn rollout，环境交互，patch evaluation 等耦合度较高的步骤。

我们将一个 SWE 任务的逻辑解耦为以下抽象：

- environment: 容器环境，可启动容器实例
- agent: 负责在已经启动的容器上，运行 sample 的 rollout；不生成 patch
- runtime (dataset?): 与数据集强相关；提供环境变量/cwd，解析 sample；生成 patch，应用 patch，运行 eval

以及以下统一类：

- sample metadata: 包括 SampleData, RewardMeta
- agent metadata: Environment, Agent, Runtime, RolloutMeta
- flow: 组合 env, agent, runtime，实现 AgentFlow

## 依赖

agentflow.base 和 environment 都是无外部依赖的；可被任何外部模块引用；
任何 .base 模块是内部依赖的，可被任何具体实现引用（如 swe.agent.minisweagent -> swe.base）
否则，必须由外部模块引用内部模块。
