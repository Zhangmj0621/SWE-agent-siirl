验证复用与评估扩展
==================

   **适合谁：** 希望通过复用训练 GPU 来提高验证吞吐量的用户。

   **你将获得：** validate-reuse 模式的配置方法与评估扩展说明。

概述
----

在 separated 模式下，siirl-agentic 可以在验证期间**复用训练 GPU** 以提高评估吞吐量。当训练处于空闲状态（等待 rollout）时，训练 GPU 临时充当额外的 rollout 引擎用于验证。

配置
----

.. code:: yaml

   trainer:
     validate_reuse_train_gpus: true        # 启用训练 GPU 复用
     validate_reuse_begin_timeout_s: 30     # 集合超时时间（秒）
     test_freq: 10                          # 每 N 步验证一次
     val_before_train: true                 # 首步训练前先运行验证

验证批次大小
------------

.. code:: yaml

   data:
     val_batch_size: null    # null = 使用完整验证集作为一个批次

   rollout:
     validate_chunk_size: 0  # 0 = 自动（train_server_concurrency × validate_workers）

注意事项
--------

- ``validate_reuse_train_gpus`` 在 colocated 模式下自动禁用
- 验证使用 ``rollout.val_kwargs`` 中的采样参数（默认贪心解码）
- 验证期间仍记录 :doc:`metrics_and_evaluation` 中的所有指标
