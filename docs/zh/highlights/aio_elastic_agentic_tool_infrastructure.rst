AIO 弹性 Agentic 工具基础设施
=============================

   **适合谁：** 需要大规模并发工具调用能力的用户。

   **你将获得：** 理解 AIO 三层分布式工具调度器如何实现弹性扩缩容和负载均衡。

核心问题
--------

Agentic RL 训练中，上千个并发 rollout 可能同时调用工具（沙箱、搜索引擎、代码执行器）。无弹性调度时，50 个工具实例面对 1000 个请求会产生级联超时。

AIO 三层架构
------------

::

   ┌──────────────────────────────────┐
   │           Proxy Layer            │
   │   (FastAPI, 请求接收 & 路由)     │
   ├──────────────────────────────────┤
   │        ResourcePool Layer        │
   │   (负载均衡: round-robin /       │
   │    least-load, 健康检查)         │
   ├──────────────────────────────────┤
   │       WorkerManager Layer        │
   │   (工具实例生命周期,              │
   │    Holt-Winters 自动扩缩)        │
   └──────────────────────────────────┘

关键特性
~~~~~~~~

================ =======================================
特性             说明
================ =======================================
**自动扩缩**     Holt-Winters 需求预测，自动增减工具实例
**负载均衡**     Round-robin 和 least-load 两种策略
**异步批量提交** 吸收延迟尖峰，防止队列爆炸
**健康检查**     自动检测和替换故障工具实例
================ =======================================

集成方式
~~~~~~~~

在 siirl-agentic 中，工具环境通过 HTTP API 与 AIO Proxy 通信：

.. code:: text

   # AIOSearchEnv 的工具调用流程
   1. POST /get_server → 获取可用工具实例
   2. 向工具实例发送请求
   3. POST /complete_task → 释放工具实例

配置示例
~~~~~~~~

.. code:: yaml

   rollout:
     multiturn:
       env_type: tool_env
       tool_env_config:
         aio_proxy_url: "http://aio-proxy:8080"
         timeout: 30

..

   详细英文版请参阅 `AIO Elastic Tool Infrastructure (English) <../../en/highlights/aio_elastic_agentic_tool_infrastructure.html>`__
