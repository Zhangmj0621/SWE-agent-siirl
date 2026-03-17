Agentic 多轮训练
================

   **适合谁：** 配置多轮 agentic rollout 与工具交互的用户。

   **你将获得：** 多轮配置详解、工具环境设置、状态机行为原理与 loss masking 说明。

概述
----

多轮 agentic 训练是 siirl-agentic 的核心差异化特性。Rollout 引擎执行一个状态机：模型生成文本、调用工具、接收环境反馈，再继续生成——全部在单个训练步内完成。

核心组件
--------

NaiveFlow 状态机
~~~~~~~~~~~~~~~~

``NaiveFlow`` 类（``siirl/execution/rollout/agent_flow/naive_flow.py``）实现多轮 rollout：

.. mermaid::

   graph LR
   A[PENDING] --> B[GENERATING]
   B --> C{有工具调用?}
   C -->|是| D[PROCESSING_ENV]
   C -->|否| E[TERMINATED]
   D --> F{达到最大轮数?}
   F -->|否| B
   F -->|是| E


AgentData（``siirl/execution/rollout/utils.py``）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每个 rollout 样本的内部状态跟踪：

.. code:: python

   class AgentData:
       def __init__(self, raw_prompt: list[dict[str, Any]]):
           self.messages = raw_prompt           # OpenAI 格式的对话历史
           self.prompts_ids = []                # 当前完整序列（prompt + 所有轮次）
           self.response_ids = []               # 当前轮次响应 token
           self.response_mask = []              # 1=模型 token，0=环境 token
           self.rollout_log_prob = []           # 所有响应 token 的 log-prob
           self.env_calls: list[FunctionCall] = []  # 待处理的工具调用
           self.env_rewards = []                # 每轮环境奖励
           self.env_turns = 0                   # 当前环境交互轮数
           self.assistant_turns = 0             # 当前 assistant 生成轮数
           self.state = AgentState.PENDING      # 当前状态机状态

配置说明
--------

完整多轮配置
~~~~~~~~~~~~

.. code:: yaml

   rollout:
     flow_function: naive               # 使用 NaiveFlow
     max_model_len: 8192                # 最大总上下文长度

     multiturn:
       env_type: tool_env               # 环境类型
       max_env_turns: 5                 # 最大工具交互轮数
       max_assistant_turns: 10          # 最大模型生成轮数
       max_parallel_calls: 4            # 每个样本的并发工具调用数
       max_env_response_length: 256     # 最大环境响应 token 数
       env_response_truncate_side: middle  # 截断方向：left/middle/right
       env_path: /path/to/env_config.yaml
       env_kwargs:
         tool_format: hermes            # 工具调用格式

   data:
     max_response_length: 4096          # 最大总响应 token 数
     mask_history: false                # 是否只在最后一轮计算 loss

参数说明
~~~~~~~~

+--------------------------------+-------------------------------+--------------------------------------+
| 参数                           | 影响                          | 推荐值                               |
+================================+===============================+======================================+
| ``max_env_turns``              | 工具交互深度                  | SWE 任务 3-10，搜索任务 1-3          |
+--------------------------------+-------------------------------+--------------------------------------+
| ``max_assistant_turns``        | 总模型生成轮数                | 设为 max_env_turns 的 2 倍           |
+--------------------------------+-------------------------------+--------------------------------------+
| ``max_parallel_calls``         | 并发工具执行数                | 1-4（越高工具吞吐越大）              |
+--------------------------------+-------------------------------+--------------------------------------+
| ``max_env_response_length``    | 工具响应截断阈值              | 根据工具类型设置 256-1024            |
+--------------------------------+-------------------------------+--------------------------------------+
| ``env_response_truncate_side`` | 截断位置                      | "middle" 保留首尾信息                |
+--------------------------------+-------------------------------+--------------------------------------+
| ``max_response_length``        | 总 token 预算                 | 所有轮次 token 之和                  |
+--------------------------------+-------------------------------+--------------------------------------+

工具调用格式
------------

siirl-agentic 通过 ``ToolParser`` 支持多种工具调用格式：

Hermes 格式（默认）
~~~~~~~~~~~~~~~~~~~~

.. code:: xml

   <tool_call>
   {"name": "search", "arguments": {"query_list": ["东京人口"]}}
   </tool_call>

GPT-OSS 格式
~~~~~~~~~~~~

.. code:: json

   {"name": "search", "arguments": "{\"query_list\": [\"东京人口\"]}"}

通过 ``env_kwargs.tool_format`` 配置：

.. code:: yaml

   rollout:
     multiturn:
       env_kwargs:
         tool_format: hermes    # 或 "gpt-oss"

Loss Masking
------------

多轮训练的关键是**正确的 loss masking**：

::

   Token 序列:   [prompt] [assistant-1] [tool-response] [assistant-2] [tool-response] [assistant-3]
   response_mask: 0 0 0    1 1 1 1 1     0 0 0 0 0       1 1 1 1       0 0 0 0 0       1 1 1 1

- **模型 token**\ （assistant 轮次）：``response_mask = 1`` → 参与策略梯度计算
- **环境 token**\ （工具响应）：``response_mask = 0`` → 从 loss 中屏蔽
- **Prompt token**：不在响应中 → 不参与 loss

这确保模型只从自身决策中学习，而非从环境输出中学习。

终止条件
--------

满足以下**任意一个**条件时，rollout 终止：

1. ``assistant_turns >= max_assistant_turns``
2. ``env_turns >= max_env_turns``
3. ``len(response_mask) >= max_response_length``
4. 模型输出中未检测到工具调用（单轮完成）
5. 环境返回 ``complete=True``

自定义工具环境
--------------

实现 ``ToolEnv`` 接口：

.. code:: python

   from siirl.environment.tool_env.base_tool_env import ToolEnv
   from siirl.environment.base import EnvResponse

   class MyCustomTool(ToolEnv):
       async def create(self, instance_id=None, **kwargs):
           # 初始化工具实例
           return instance_id, EnvResponse()

       async def step(self, action: dict) -> EnvResponse:
           # 执行工具动作
           result = await my_tool_logic(action)
           return EnvResponse(
               text=result,
               rewards=0.5,       # 可选的逐步奖励
               complete=False,    # 设为 True 则终止轨迹
           )

       async def release(self, instance_id: str):
           # 清理工具实例
           pass

常见问题
--------

+-----------------------------------------------------+---------------------------+----------------------------------------+
| 现象                                                | 原因                      | 解决方案                               |
+=====================================================+===========================+========================================+
| ``max_response_length`` 小于多轮总 token 数         | 过早截断                  | 增大到 4096-8192                       |
+-----------------------------------------------------+---------------------------+----------------------------------------+
| SWE 任务中 ``max_env_turns=1``                      | 只有一次工具调用          | 增大到 5+                              |
+-----------------------------------------------------+---------------------------+----------------------------------------+
| 缺少 ``env_path``                                   | 工具环境未加载            | 提供工具配置文件路径                   |
+-----------------------------------------------------+---------------------------+----------------------------------------+
| ``tool_format`` 不匹配                              | 工具调用未被解析          | 与模型训练数据的格式保持一致           |
+-----------------------------------------------------+---------------------------+----------------------------------------+
| ``max_parallel_calls`` 过高                         | 工具服务器过载            | 从 1 开始逐步增大                      |
+-----------------------------------------------------+---------------------------+----------------------------------------+
