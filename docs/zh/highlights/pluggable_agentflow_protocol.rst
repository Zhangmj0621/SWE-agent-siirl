可插拔 AgentFlow 协议
=====================

   **适合谁：** 需要定义自定义 Agentic 任务类型的用户和开发者。

   **你将获得：** 理解 AgentFlow 的三阶段协议以及如何通过 YAML 配置注入实现零代码任务切换。

三阶段协议
----------

.. code:: python

   class AgentFlow(Protocol):
       def preprocess(self, data: dict) -> list[Sample]:
           """将原始数据转换为 Sample 对象"""

       def generate(self, model: Model, samples: list[Sample]) -> list[Sample]:
           """运行模型推理和环境交互"""

       def reward(self, samples: list[Sample]) -> list[Sample]:
           """计算完成轨迹的奖励"""

核心优势
~~~~~~~~

1. **配置驱动** — 通过 YAML 指定 flow 类和自定义函数，无需修改框架代码。

2. **动态函数注入** — ``load_agentflow()`` 使用 ``MethodType`` 将自定义的 ``preprocess_fn``\ 、\ ``generate_fn``\ 、\ ``reward_fn`` 注入到 flow 实例上。

3. **内置流程** — 提供 ``swe`` 等内置流程，用户也可以用 Python 模块路径引用自定义流程。

配置示例
~~~~~~~~

.. code:: yaml

   rollout:
     agentflow: "swe"                                    # 内置 SWE 流程
     agentflow_preprocess_fn: "my_project:preprocess"    # 自定义预处理
     agentflow_reward_fn: "my_project:custom_reward"     # 自定义奖励

关键文件
~~~~~~~~

+---------------------------------------------------+------------------------------------+
| 文件                                              | 描述                               |
+===================================================+====================================+
| ``siirl/execution/rollout/agentflow/base.py``     | Protocol 定义、Sample/Model 数据类 |
+---------------------------------------------------+------------------------------------+
| ``siirl/execution/rollout/agentflow/__init__.py`` | ``load_agentflow()`` 动态加载      |
+---------------------------------------------------+------------------------------------+

..

   详细英文版请参阅 `Pluggable AgentFlow Protocol (English) <../../en/highlights/pluggable_agentflow_protocol.html>`__
