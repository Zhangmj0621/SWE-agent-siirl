性能调优
========

   **适合谁：** 需要优化训练吞吐量和 GPU 利用率的用户。

   **你将获得：** 异步流水线、rollout 并发度和内存管理的调优策略。

流水线吞吐量
------------

异步因子（async_factor）
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   trainer:
     async_factor: 2   # 预缓冲 2 个 rollout 批次

``async_factor`` 越大，rollout 和训练的解耦程度越高，但 off-policy 数据陈旧度也越高。

Off-Policy 训练
~~~~~~~~~~~~~~~

.. code:: yaml

   trainer:
     off_policy_step: 2       # 接受 [当前-2, 当前] 版本范围内的数据
     off_policy_strategy: fifo

允许在略微陈旧的数据上训练，在长 agentic rollout 期间最大化 GPU 利用率。

Rollout 并发度
--------------

.. code:: yaml

   rollout:
     train_server_concurrency: 256    # 每个引擎的并发请求数
     max_num_seqs: 0                  # 0 = 自动（4 × 并发数）

对于含工具调用的 agentic 任务，增大并发度可在单个请求等待工具响应时保持引擎繁忙。

内存优化
--------

参数卸载
~~~~~~~~

.. code:: yaml

   actor_ref:
     actor:
       megatron:
         param_offload: true       # 参数卸载到 CPU
         grad_offload: true        # 梯度卸载到 CPU
         optimizer_offload: true   # 优化器状态卸载到 CPU

动态批处理
~~~~~~~~~~

.. code:: yaml

   actor_ref:
     actor:
       use_dynamic_batch: true       # 基于 token 数量的动态批处理
       max_tokens_per_gpu: 4096      # 每 GPU 最大 token 数
       use_workload_balance: true    # 基于 FLOPs 的负载均衡

多节点扩展
----------

.. code:: yaml

   trainer:
     nnodes: 4
     n_gpus_per_node: 8
     actor_gpus: 16    # 2 个节点用于训练
     rollout_gpus: 16  # 2 个节点用于 rollout

调优建议
--------

.. list-table::
   :header-rows: 1

   * - 瓶颈
     - 诊断方法
     - 解决方案
   * - GPU 利用率低（工具等待）
     - ``nvidia-smi`` 显示 <50%
     - 增大 ``async_factor``，启用 off-policy
   * - 训练慢（显存不足）
     - OOM 报错
     - 启用参数卸载
   * - Rollout 吞吐低
     - 日志显示 generation_duration 高
     - 增大 ``train_server_concurrency``
   * - 参数同步慢
     - 训练步之间停顿明显
     - 增大 ``param_sync_buffer_size``
