代码结构
========

   **适合谁：** 需要浏览和修改 siirl-agentic 代码库的框架贡献者。

   **你将获得：** 代码组织方式、调用链路与扩展点的开发者向导。

入口点
------

一切从 ``siirl/async_train.py`` 开始：

.. code:: python

   # MainRunner 是一个 Ray Actor，负责编排整个训练生命周期
   @ray.remote
   class MainRunner:
       def run(self):
           # 阶段 1：解析 YAML 配置 → SiiRLArguments
           # 阶段 2：分配 GPU 资源（actor_gpus, rollout_gpus, critic_gpus）
           # 阶段 3：初始化 DataCoordinator（DataBuffer + DataLoader）
           # 阶段 4：初始化组件（Trainer, RolloutManager, Critic）
           # 阶段 5：启动异步训练循环
           # 阶段 6：等待完成
           # 阶段 7：通过 TaskCoordinator 检查状态
           # 阶段 8：清理

训练步调用链
------------

::

   MainRunner.run()
     └─► Trainer.train()           # siirl/worker/actor/
           ├─► DataBuffer.get()    # siirl/data_coordinator/data_buffer.py
           ├─► Actor.forward()     # siirl/engine/actor/
           ├─► Critic.forward()    # （仅 PPO）
           ├─► compute_advantage() # siirl/algorithm/advantage.py
           ├─► compute_loss()      # siirl/algorithm/loss.py
           └─► Actor.backward()    # 梯度更新

Rollout 步调用链
----------------

::

   RolloutManager.rollout()          # siirl/worker/rollout/
     └─► AgentExecutor.execute()     # siirl/execution/rollout/agent_executor/
           └─► NaiveFlow.run()       # siirl/execution/rollout/agent_flow/naive_flow.py
                 ├─► preprocess()    # AgentFlow 协议
                 ├─► generate()      # SGLang 推理
                 ├─► Environment.step()  # 工具执行
                 ├─► （重复直到终止）
                 └─► reward()        # 计算轨迹奖励

AgentFlow 加载链
----------------

::

   load_agentflow(config)            # siirl/execution/rollout/agentflow/__init__.py
     ├─► 加载 AgentFlow 类（如 NaiveFlow）
     ├─► 从配置导入 preprocess_fn
     ├─► 从配置导入 generate_fn
     ├─► 从配置导入 reward_fn
     └─► 通过 MethodType 注入到 flow 实例

配置流转
--------

::

   YAML 文件
     └─► parser.py → SiiRLArguments     # siirl/params/parser.py
           ├─► .data: DataArguments       # siirl/params/data_args.py
           ├─► .actor_ref: ActorArguments # siirl/params/model_args.py
           ├─► .rollout: RolloutArguments # siirl/params/model_args.py
           ├─► .critic: CriticArguments   # siirl/params/model_args.py
           └─► .trainer: TrainingArguments # siirl/params/training_args.py

关键扩展点
----------

1. 新增 AgentFlow
~~~~~~~~~~~~~~~~~

.. code:: python

   # siirl/execution/rollout/agentflow/my_flow.py
   from .base import AgentFlow, Sample, Model

   class MyFlow:
       def preprocess(self, data: dict) -> list[Sample]: ...
       def generate(self, model: Model, samples: list[Sample]) -> list[Sample]: ...
       def reward(self, samples: list[Sample]) -> list[Sample]: ...

在 YAML 中引用：

.. code:: yaml

   rollout:
     agentflow: "siirl.execution.rollout.agentflow.my_flow:MyFlow"

2. 新增环境
~~~~~~~~~~~

.. code:: python

   # siirl/environment/my_env.py
   from siirl.environment.base import BasEnvironment, EnvResponse

   class MyEnvironment(BasEnvironment):
       async def step(self, action: str) -> EnvResponse: ...
       async def reset(self) -> EnvResponse: ...

3. 新增奖励函数
~~~~~~~~~~~~~~~

.. code:: python

   # my_rewards.py
   def custom_reward(samples):
       for sample in samples:
           sample.reward = compute_reward(sample.conversations)
       return samples

.. code:: yaml

   custom_reward_function: "my_rewards:custom_reward"

测试结构
--------

::

   tests/
   ├── unit/            # 快速单元测试
   ├── integration/     # 多组件集成测试
   ├── e2e/             # 端到端训练测试（需要 GPU）
   └── conftest.py      # 共享 fixture

运行测试：

.. code:: bash

   pytest tests/unit/              # 快速单元测试
   pytest tests/integration/       # 集成测试
   pytest -m gpu                   # 依赖 GPU 的测试

相关文档
--------

- :doc:`模块地图 <../reference/module_map>` — 快速目录参考
- :doc:`新增 Executor 或 Flow <adding_new_executor_or_flow>` — 分步教程
- :doc:`架构概览 <../concepts/architecture_overview>` — 高层设计
