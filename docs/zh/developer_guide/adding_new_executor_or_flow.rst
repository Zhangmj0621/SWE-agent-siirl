添加新 Executor 或 AgentFlow
============================

   **适合谁：** 需要为 siirl-agentic 扩展新任务类型或 rollout 策略的框架贡献者。

   **你将获得：** 实现并注册自定义 AgentFlow 的完整分步教程。

前置条件
--------

- 熟悉 :doc:`AgentFlow 协议 <../highlights/pluggable_agentflow_protocol>`
- 了解 :doc:`架构概览 <../concepts/architecture_overview>`
- 本地开发环境已搭建（:doc:`安装 <../get_started/installation>`）

AgentFlow 协议
--------------

每个 AgentFlow 必须实现三个阶段：

.. code:: python

   class AgentFlow(Protocol):
       def preprocess(self, data: dict) -> list[Sample]:
           """将原始数据转换为 Sample 对象。"""
           ...

       def generate(self, model: Model, samples: list[Sample]) -> list[Sample]:
           """运行模型推理和环境交互。"""
           ...

       def reward(self, samples: list[Sample]) -> list[Sample]:
           """为完成的轨迹计算奖励。"""
           ...

``Sample`` 数据类的关键字段：

+-----------------------+----------------+----------------------------------------------------------+
| 字段                  | 类型           | 说明                                                     |
+=======================+================+==========================================================+
| ``tokens``            | ``Tensor``     | 完整 token 序列（prompt + response）                     |
+-----------------------+----------------+----------------------------------------------------------+
| ``loss_mask``         | ``Tensor``     | 1=参与训练的模型 token，0=环境/prompt token              |
+-----------------------+----------------+----------------------------------------------------------+
| ``rollout_log_probs`` | ``Tensor``     | rollout 策略的 log-probability                           |
+-----------------------+----------------+----------------------------------------------------------+
| ``conversations``     | ``list[dict]`` | 完整对话历史                                             |
+-----------------------+----------------+----------------------------------------------------------+
| ``reward``            | ``float``      | 轨迹的标量奖励                                           |
+-----------------------+----------------+----------------------------------------------------------+

第一步：创建 Flow 模块
----------------------

在 ``siirl/execution/rollout/agentflow/`` 中创建新文件：

.. code:: python

   # siirl/execution/rollout/agentflow/my_task_flow.py
   from siirl.execution.rollout.agentflow.base import AgentFlow, Sample, Model

   class MyTaskFlow:
       """
       自定义 AgentFlow，用于 [描述你的任务类型]。
       实现三阶段 AgentFlow 协议：preprocess → generate → reward
       """

       def __init__(self, config=None):
           self.config = config or {}

       def preprocess(self, data: dict) -> list[Sample]:
           samples = []
           for item in data['items']:
               sample = Sample(
                   conversations=[{"role": "user", "content": item['prompt']}],
               )
               samples.append(sample)
           return samples

       def generate(self, model: Model, samples: list[Sample]) -> list[Sample]:
           for sample in samples:
               response = model.generate(sample.conversations)
               sample.tokens = response.tokens
               sample.rollout_log_probs = response.log_probs
               sample.loss_mask = response.loss_mask
               sample.conversations.append({
                   "role": "assistant",
                   "content": response.text
               })
           return samples

       def reward(self, samples: list[Sample]) -> list[Sample]:
           for sample in samples:
               sample.reward = self._evaluate(sample)
           return samples

       def _evaluate(self, sample: Sample) -> float:
           # 实现你的奖励逻辑
           return 1.0

第二步：注册 Flow
-----------------

方式 A：YAML 配置直接引用（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在训练配置中直接引用：

.. code:: yaml

   rollout:
     agentflow: "siirl.execution.rollout.agentflow.my_task_flow:MyTaskFlow"

``load_agentflow()`` 会动态导入并实例化该类。

方式 B：添加到内置注册表
~~~~~~~~~~~~~~~~~~~~~~~~

在 ``siirl/execution/rollout/agentflow/__init__.py`` 中添加条目：

.. code:: python

   BUILTIN_FLOW = {
       "swe": ".swe:agentflow",
       "my_task": ".my_task_flow:MyTaskFlow",  # 添加这一行
   }

然后通过别名引用：

.. code:: yaml

   rollout:
     agentflow: "my_task"

第三步：函数注入（可选）
------------------------

AgentFlow 支持动态函数注入，无需继承，直接通过配置覆盖单个阶段：

.. code:: yaml

   rollout:
     agentflow: "swe"                    # 使用内置 SWE flow
     agentflow_preprocess_fn: "my_project.preprocess:custom_preprocess"
     agentflow_reward_fn: "my_project.rewards:custom_reward"

注入函数通过 ``MethodType`` 绑定到 flow 实例：

.. code:: python

   # my_project/rewards.py
   def custom_reward(self, samples):
       """'self' 是 AgentFlow 实例，可访问 self.config 等。"""
       for sample in samples:
           sample.reward = my_scoring_function(sample.conversations)
       return samples

第四步：实现多轮 Flow
---------------------

对于需要环境交互的任务：

.. code:: python

   def generate(self, model: Model, samples: list[Sample]) -> list[Sample]:
       for sample in samples:
           for turn in range(self.config.get('max_turns', 5)):
               # 1. 生成模型响应
               response = model.generate(sample.conversations)

               # 2. 检查工具调用
               if not response.has_tool_call:
                   break

               # 3. 通过环境执行工具调用
               env_response = self.environment.step(response.tool_call)

               # 4. 追加到对话
               sample.conversations.extend([
                   {"role": "assistant", "content": response.text},
                   {"role": "tool", "content": env_response.text},
               ])

               # 5. 更新 loss mask（屏蔽环境 token）
               sample.loss_mask = self._update_loss_mask(
                   sample.loss_mask, response, env_response
               )
       return samples

第五步：编写测试
----------------

.. code:: python

   # tests/unit/test_my_task_flow.py
   from siirl.execution.rollout.agentflow.my_task_flow import MyTaskFlow

   class TestMyTaskFlow:
       def test_preprocess(self):
           flow = MyTaskFlow()
           data = {"items": [{"prompt": "Hello"}]}
           samples = flow.preprocess(data)
           assert len(samples) == 1
           assert samples[0].conversations[0]["content"] == "Hello"

       def test_reward(self):
           flow = MyTaskFlow()
           samples = [Sample(conversations=[...])]
           results = flow.reward(samples)
           assert all(s.reward is not None for s in results)

.. code:: bash

   pytest tests/unit/test_my_task_flow.py -v

完成 Checklist
--------------

- ☐ Flow 实现了全部三个协议方法（``preprocess``、``generate``、``reward``）
- ☐ ``preprocess`` 返回正确初始化的 ``Sample`` 对象
- ☐ ``generate`` 填充了 ``tokens``、``rollout_log_probs`` 和 ``loss_mask``
- ☐ ``reward`` 为每个样本赋值了标量 ``reward``
- ☐ Flow 已注册（YAML 路径或内置注册表）
- ☐ 单元测试覆盖了三个阶段
- ☐ 如果添加了内置 flow，文档已更新

相关文档
--------

- :doc:`可插拔 AgentFlow 协议 <../highlights/pluggable_agentflow_protocol>` — 设计原理
- :doc:`代码结构 <code_structure>` — 完整代码库向导
- :doc:`Agentic 多轮训练 <../user_guide/agentic_multiturn>` — 多轮配置
