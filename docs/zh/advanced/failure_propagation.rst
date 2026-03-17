故障传播与优雅关停
==================

   **适合谁：** 运行大规模分布式训练、需要了解故障处理机制的用户。

   **你将获得：** 故障如何在分布式组件之间检测、传播和处理的完整说明。

概述
----

在分布式 MPMD 系统中，某个组件的故障（如 Trainer 的 CUDA OOM）必须传播到所有其他组件，以防止僵尸进程和资源浪费。siirl-agentic 使用集中式的 ``TaskCoordinator`` Ray Actor 进行生命周期管理。

TaskCoordinator
---------------

``TaskCoordinator``（``siirl/utils/task_coordinator.py``）是一个 Ray Actor，提供：

- 面向所有组件的\ **统一停止信号**
- 跨 Trainer、RolloutManager 和 DataCoordinator 的\ **故障传播**
- **优雅关停**\ 协调
- 用于事后调试的\ **事件日志**

生命周期状态
~~~~~~~~~~~~

::

   RUNNING  ──► COMPLETED   （正常完成）
      │
      ├──────► FAILED       （任意组件出错）
      │
      └──────► SHUTDOWN     （请求优雅关停）

所有非 RUNNING 状态都会使 ``should_stop()`` 返回 ``True``。

组件如何使用 Coordinator
------------------------

1. 初始化
~~~~~~~~~

Coordinator 在 ``MainRunner`` 中创建，并传递给所有组件：

.. code:: python

   coordinator = TaskCoordinator.remote()
   trainer = Trainer(..., coordinator=coordinator)
   rollout_manager = RolloutManager(..., coordinator=coordinator)

2. 轮询停止信号
~~~~~~~~~~~~~~~

每个组件在主循环中定期检查 ``should_stop()``：

.. code:: python

   while True:
       if ray.get(self.coordinator.should_stop.remote()):
           logger.info("Stop signal received, exiting...")
           break
       # ... 继续工作 ...

3. 上报故障
~~~~~~~~~~~

当组件遇到错误时，向 coordinator 上报：

.. code:: python

   try:
       self.train_step(batch)
   except Exception as e:
       ray.get(self.coordinator.report_failure.remote(
           source="trainer_0",
           reason=str(e)
       ))
       raise

4. 请求关停
~~~~~~~~~~~

正常完成时（如所有 epoch 训练结束）：

.. code:: python

   ray.get(self.coordinator.request_shutdown.remote(
       source="main_runner",
       reason="Training completed"
   ))

典型故障场景
------------

Trainer CUDA OOM
~~~~~~~~~~~~~~~~

1. Trainer 捕获 ``RuntimeError: CUDA out of memory``
2. 调用 ``coordinator.report_failure.remote("trainer", "CUDA OOM")``
3. Coordinator 将状态设置为 ``FAILED``
4. RolloutManager 下次调用 ``should_stop()`` 时返回 ``True``
5. RolloutManager 取消待处理的 rollout 并退出
6. ``MainRunner`` 通过 ``coordinator.get_summary()`` 检测到故障并执行清理

Rollout 超时
~~~~~~~~~~~~

1. RolloutManager 检测到 rollout 超过最大时间
2. 上报含超时详情的故障
3. Trainer 停止从 DataBuffer 消费
4. 所有组件优雅关停

Ray Actor 崩溃
~~~~~~~~~~~~~~

如果 Ray Actor 在未上报的情况下崩溃：

1. ``MainRunner`` 通过 Ray 内置的 actor 故障检测监控 actor 健康状态
2. 检测到 actor 死亡后，调用 ``coordinator.report_failure.remote()``
3. 剩余 actor 在下次轮询时收到停止信号

调试故障
--------

事件日志
~~~~~~~~

获取完整事件历史以供事后分析：

.. code:: python

   events = ray.get(coordinator.get_events.remote())
   for event in events:
       print(f"[{event['timestamp']}] {event['source']}: {event['message']}")

状态摘要
~~~~~~~~

快速了解协调状态：

.. code:: python

   summary = ray.get(coordinator.get_summary.remote())
   # {
   #     "status": "failed",
   #     "failure_reason": "CUDA OOM in trainer_0",
   #     "duration_seconds": 3421.5,
   #     "event_count": 47
   # }

最佳实践
--------

1. **轮询频率**：至少在每个训练步或 rollout 批次检查一次 ``should_stop()``，不要在紧内循环中检查
2. **务必上报**：在组件边界捕获异常并上报后再重新抛出
3. **清理资源**：使用 try/finally 块确保 GPU 内存、Ray 对象引用和临时文件被清理
4. **记录上下文**：在故障上报中包含组件名称、GPU rank 和错误详情

相关文档
--------

- :doc:`架构概览 <../concepts/architecture_overview>` — MPMD 组件角色
- :doc:`性能调优 <performance_tuning>` — 通过卸载和批处理预防 OOM
- :doc:`常见问题排障 <../faq/troubleshooting>` — 常见故障模式与解决方案
