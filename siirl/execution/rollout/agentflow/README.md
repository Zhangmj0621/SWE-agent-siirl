# AgentFlow 文档

AgentFlow 是一个用于构建和管理 AI Agent 工作流的框架。它提供了标准化的接口来处理数据预处理、模型生成和结果评估。

## 核心概念

基础组件：

- **AgentFlow**: 基础工作流类
- **Sample**: 数据样本容器,包含状态、对话历史、token 信息等
- **Model**: 模型接口抽象
- **ModelResponse**: 模型响应封装

标准工作流：

AgentFlow 定义了三个核心方法:

1. **preprocess** (`Callable[[AgentFlow, dict], Sample]`): 数据预处理,将原始数据转换为 Sample 对象；
2. **generate** (`Callable[[AgentFlow, Sample], Awaitable[None]]`): 调用模型生成响应
3. **reward** (`Callable[[AgentFlow, Sample], Awaitable[None]]`): 评估生成结果并计算奖励

我们实现了动态注入机制，所有方法函数都可以通过配置文件进行修改注入。

例如，可以定义独立的奖励函数并通过配置注入:

```python
async def eval_reward(_: AgentFlow, sample: Sample):
    sample.reward = 1.0

agent = load_agentflow(
    {
        "agent_name": "example.agentflow:SimpleAgentFlow",  # 目标 AgentFlow 的构造函数，或是预定义的 alias
        "reward_fn": "example.agentflow:eval_reward",       # 设置覆盖的 reward function
    },
    model,
)
```

## 实践约定

1. **导入约定**: 避免依赖父模块（注意同路径下的 `__init__` 也是父模块），引入它会导致循环引入；请使用 `.base`；跨模块注意避免循环。
2. **异步方法**: `generate` 和 `reward` 必须是异步方法
3. **就地修改**: 大部分方法直接修改 `sample` 对象，不返回新对象
4. **状态管理**: 正确设置 `sample.status` 标识处理状态
5. **错误处理**: AgentFlow 仅初始化一次可以抛出异常；Agent 中应捕获异常并设置 sample 状态和相关 log
6. **动态数据存储**: 尽量使用 dataclass 显式声明数据，而非 dict 纯动态存储
7. **Sample token 更新**: 每次 query 前后调用 `sample.append_xxx` 更新 sample token。

## TODO

- 使用 Pydantic 解析配置
- 启用动态加载的 alias
- 增加异常处理
- 换名字
