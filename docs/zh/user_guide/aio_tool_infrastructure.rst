AIO 工具基础设施
================

   **适合谁：** 需要部署和配置 AIO 进行 agentic 工具管理的用户。

   **你将获得：** 部署 AIO Proxy、WorkerManager 和配置自动扩缩的分步指南。

概述
----

AIO（Agentic I/O）是一个分布式工具调度基础设施，负责在异构节点上管理外部工具实例。架构原理详见 :doc:`AIO 亮点页 <../highlights/aio_elastic_agentic_tool_infrastructure>`。

三层架构
--------

.. code:: text

   客户端（siirl-agentic Rollout）
     │
     ▼
   Proxy（代理层）           ← 统一入口，路由请求
     │
     ▼
   ResourcePool（资源池层）   ← 跟踪容量，负载均衡
     │
     ▼
   WorkerManager（工作节点）  ← 管理工具实例，执行工具调用

部署步骤
--------

第一步：启动 AIO Proxy
~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python -m aio.Scheduler.proxy --config aio_config.yaml

第二步：启动 WorkerManager
~~~~~~~~~~~~~~~~~~~~~~~~~~

在每个工具节点上运行：

.. code:: bash

   python -m aio.Scheduler.Resources.worker_manager \
       --proxy-url http://proxy-host:8080

第三步：验证注册状态
~~~~~~~~~~~~~~~~~~~~

WorkerManager 启动时会自动向 Proxy 注册工具。确认注册成功：

.. code:: bash

   curl http://proxy-host:8080/status

AIO 配置
--------

.. code:: yaml

   aio:
     proxy:
       host: 0.0.0.0
       port: 8080
     tools:
       - name: search
         capacity_per_worker: 10     # 每个 worker 的并发容量
       - name: sandbox_fusion
         capacity_per_worker: 5
     auto_scaling:
       enabled: true
       algorithm: holt_winters        # 使用 Holt-Winters 时序预测

与 rollout 集成
---------------

.. code:: yaml

   rollout:
     multiturn:
       env_type: tool_env
       env_kwargs:
         tool_format: hermes
         aio_proxy_url: http://proxy-host:8080

自动扩缩原理
------------

AIO 使用 **Holt-Winters 时间序列预测器**\ 监控每步并发分布：

1. ``ConcurrencyMonitor`` 持续采样当前并发工具调用数
2. Holt-Winters 预测下一时间窗口的需求峰值
3. 当预测需求超过当前容量时，自动激活更多工具环境
4. 当负载降低时，空闲实例被回收以释放资源

监控
----

- ``/status`` — 当前资源池状态
- ``/metrics`` — 并发度与延迟指标
- 查看 ``ConcurrencyMonitor`` 日志了解需求模式

常见问题
--------

+-------------------------------------------+---------------------------+-------------------------------------------+
| 现象                                      | 原因                      | 解决方案                                  |
+===========================================+===========================+===========================================+
| ``Error getting server from master node`` | AIO Proxy 未运行          | 启动 ``python -m aio.Scheduler.proxy``    |
+-------------------------------------------+---------------------------+-------------------------------------------+
| 工具调用超时                              | 工具实例过载              | 增加工具节点或增大 capacity_per_worker    |
+-------------------------------------------+---------------------------+-------------------------------------------+
| 工具未注册                                | WorkerManager 未启动      | 在每个工具节点上启动 WorkerManager        |
+-------------------------------------------+---------------------------+-------------------------------------------+
