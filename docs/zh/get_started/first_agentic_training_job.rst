首个 Agentic 训练任务
=====================

   **适合谁：** 想要训练具备多轮工具交互能力的 LLM Agent 的用户。

   **你将获得：** 一个完整运行的 GRPO 训练任务，其中 agent 在 rollout 中使用搜索工具。

前置条件
--------

- siirl-agentic 已安装并验证（参见 :doc:`安装 <installation>`）
- 已完成 :doc:`快速上手 <quickstart>`\ （基础 GRPO 训练可正常运行）
- **AIO 工具基础设施** 已克隆并安装。AIO 是 monorepo 中的独立仓库：

  .. code:: bash

     # 从 monorepo 根目录（siirl-agentic/ 的上级目录）
     cd AIO
     pip install -e .

什么是"Agentic"训练？
---------------------

在标准 RL 训练中，模型生成单个响应并获得奖励。在 **agentic** 训练中：

1. 模型生成的文本可能包含**工具调用**（如搜索查询、代码执行）
2. 工具调用通过 AIO 基础设施在**真实环境中执行**
3. 工具响应作为环境观测**追加到对话中**
4. 模型基于工具响应**继续生成**
5. 这个多轮循环重复直到终止（达到最大轮数或模型主动结束）
6. **只有模型生成的 token** 参与策略梯度（环境 token 被屏蔽）

第一步：配置多轮 Rollout
------------------------

在训练配置中添加多轮配置：

.. code:: yaml

   rollout:
     flow_function: naive
     multiturn:
       env_type: tool_env
       max_env_turns: 5
       max_assistant_turns: 10
       max_parallel_calls: 4
       max_env_response_length: 256
       env_response_truncate_side: middle
       env_path: /path/to/tool_env_config.yaml
       env_kwargs:
         tool_format: hermes

第二步：配置工具环境
--------------------

创建工具环境配置文件（``tool_env_config.yaml``）：

.. code:: yaml

   tools:
     - name: search
       type: aio_search
       config:
         topk: 3

第三步：启动 AIO 基础设施
--------------------------

训练开始前，先启动 AIO Proxy 和 WorkerManager：

.. code:: bash

   # 启动 AIO Proxy
   python -m aio.Scheduler.proxy --config aio_config.yaml

   # 在每个工具节点启动 WorkerManager
   python -m aio.Scheduler.Resources.worker_manager --proxy-url http://proxy-host:8080

第四步：启动 Agentic 训练
--------------------------

.. code:: bash

   cd siirl-agentic
   bash examples/AIO/run_qwen3_8b.sh

**预期输出：**

::

   INFO  | Ray is initialized. Time cost: 150.23 ms
   INFO  | MainRunner started. Beginning workflow setup...
   INFO  | Initializing DataCoordinator with 1 distributed DataBuffers...
   SUCCESS | RolloutManager initialized. Router at: http://10.0.0.1:30000
   INFO  | Starting async training loop...
   INFO  | [NaiveFlow] PENDING -> GENERATING (sample 0)
   INFO  | [NaiveFlow] GENERATING -> PROCESSING_ENV (sample 0, 2 tool calls)
   INFO  | [AIOSearchTool] search query dispatched to worker
   INFO  | [NaiveFlow] PROCESSING_ENV -> GENERATING (sample 0, env_turn 1)
   INFO  | [NaiveFlow] GENERATING -> TERMINATED (sample 0, reward=1.0)

第五步：监控训练
----------------

关键指标：

- **reward/mean** — 每步平均奖励（应逐步上升）
- **rollout/generation_duration** — LLM 生成耗时
- **rollout/reward_duration** — 奖励计算耗时
- **rollout/env_turns_mean** — 每个样本的平均工具交互轮数

理解轨迹结构
------------

一个典型的 agentic 轨迹如下：

::

   [用户]    东京的人口是多少？
   [助手]    我来搜索一下这个信息。
             <tool_call>search(query_list=["东京人口"])</tool_call>
   [工具]    东京人口约为 1396 万……
   [助手]    根据搜索结果，东京人口约为 1396 万人。

Token 视角：

- 用户 prompt → ``response_mask = 0``\ （不参与训练）
- 助手轮次 1 → ``response_mask = 1``\ （参与训练）
- 工具响应 → ``response_mask = 0``\ （从 loss 中屏蔽）
- 助手轮次 2 → ``response_mask = 1``\ （参与训练）

常见问题
--------

+-----------------------------------------+-------------------------------------------+-----------------------------------------+
| 现象                                    | 原因                                      | 解决方案                                |
+=========================================+===========================================+=========================================+
| ``AIOSearchTool: Error getting server`` | AIO Proxy 未启动                          | 运行 ``python -m aio.Scheduler.proxy``  |
+-----------------------------------------+-------------------------------------------+-----------------------------------------+
| ``TERMINATED after 1 turn``             | ``max_assistant_turns=1``                 | 增大 ``max_assistant_turns``            |
+-----------------------------------------+-------------------------------------------+-----------------------------------------+
| 工具响应被截断                          | ``max_env_response_length`` 过小          | 增大该值                                |
+-----------------------------------------+-------------------------------------------+-----------------------------------------+
| 所有奖励为 0                            | 奖励函数不兼容多轮轨迹                    | 检查自定义奖励函数逻辑                  |
+-----------------------------------------+-------------------------------------------+-----------------------------------------+

下一步
------

- :doc:`Agentic 多轮训练指南 <../user_guide/agentic_multiturn>` — 多轮配置深度解析
- :doc:`AIO 工具基础设施 <../user_guide/aio_tool_infrastructure>` — 配置与扩展工具环境
- :doc:`可插拔 AgentFlow <../highlights/pluggable_agentflow_protocol>` — 自定义 rollout flow
