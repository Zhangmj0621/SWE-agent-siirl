工具环境与 SWE 任务
===================

   **适合谁：** 为 SWE-bench 和类似编程任务构建或配置工具环境的用户。

   **你将获得：** 工具环境配置方法、自定义工具实现指南与 SWE 风格训练配置。

概述
----

siirl-agentic 支持在 SWE 风格任务上进行 agentic 训练，模型与代码沙箱、文件系统和测试运行器进行交互。工具环境系统为这些交互提供统一接口。

工具环境架构
------------

.. mermaid::

   graph TB
   A[NaiveFlow] -->|tool_call| B[ToolEnv Manager]
   B --> C[ToolParser - 提取调用]
   C --> D{工具类型}
   D -->|search| E[AIOSearchEnv]
   D -->|sandbox| F[SandboxEnv]
   D -->|custom| G[CustomToolEnv]


内置工具环境
------------

AIO 搜索环境
~~~~~~~~~~~~

用于检索增强型任务：

.. code:: yaml

   rollout:
     multiturn:
       env_type: tool_env
       env_kwargs:
         tool_format: hermes

关键文件：``siirl/environment/tool_env/aio_search_env.py``

自定义工具环境
~~~~~~~~~~~~~~

继承 ``ToolEnv`` 实现自定义工具：

.. code:: python

   from siirl.environment.tool_env.base_tool_env import ToolEnv
   from siirl.environment.base import EnvResponse

   class MySWETool(ToolEnv):
       async def step(self, action: dict) -> EnvResponse:
           # 在沙箱中运行代码，执行测试等
           result = await self.run_in_sandbox(action["code"])
           return EnvResponse(
               text=result.output,
               rewards=1.0 if result.tests_passed else 0.0,
               complete=result.all_tests_passed,
           )

SWE-Bench 配置
--------------

.. code:: yaml

   rollout:
     flow_function: naive
     multiturn:
       env_type: tool_env
       max_env_turns: 10
       max_assistant_turns: 20
       max_env_response_length: 1024
       env_kwargs:
         tool_format: hermes

   data:
     max_response_length: 16384   # SWE 任务需要长上下文

常见问题
--------

+-------------------------------+---------------------------+-------------------------------+
| 现象                          | 原因                      | 解决方案                      |
+===============================+===========================+===============================+
| 沙箱不可访问                  | 工具执行超时              | 检查 AIO Proxy 连接状态       |
+-------------------------------+---------------------------+-------------------------------+
| ``max_response_length`` 过小  | 编辑不完整                | 增大到 16384+                 |
+-------------------------------+---------------------------+-------------------------------+
| ``tool_format`` 不匹配        | 工具调用未被解析          | 与模型期望的格式保持一致      |
+-------------------------------+---------------------------+-------------------------------+
