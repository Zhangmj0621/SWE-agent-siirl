部署模式
========

   **适合谁：** 需要在 separated 和 colocated GPU 部署之间做选择的用户。

   **你将获得：** 每种部署拓扑的说明、适用场景与配置方法。

Separated 模式（默认）
----------------------

GPU 分别分配给训练和 rollout，各组拥有专用资源。

.. code:: yaml

   trainer:
     colocate: false
     actor_gpus: 4
     rollout_gpus: 4

**适用场景：** 大多数生产训练。资源边界清晰，性能可预测。

Colocated 模式
--------------

训练和 rollout 通过权重卸载共享全部 GPU。

.. code:: yaml

   trainer:
     colocate: true

**适用场景：** GPU 资源有限、模型较小的场景。框架自动管理：

- ``megatron.param_offload = true``
- ``rollout.gpu_memory_utilization`` 限制在 0.45
- ``validate_reuse_train_gpus`` 自动禁用

对比
----

=============== ======================== ==========================
维度            Separated                Colocated
=============== ======================== ==========================
GPU 效率        各角色独占               时分复用
显存压力        较低                     较高
复杂度          简单                     需要管理卸载
最适合          生产环境、大模型         原型开发、小模型
=============== ======================== ==========================
