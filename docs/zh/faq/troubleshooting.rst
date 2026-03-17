常见问题与排障
==============

   **适合谁：** 训练过程中遇到问题的用户。

   **你将获得：** 最常见故障模式的解决方案。

初始化故障
----------

Ray 连接问题
~~~~~~~~~~~~

**现象：** ``ConnectionError: Ray is not initialized``

**根因：** Ray 集群未启动或存在冲突的 Ray 进程。

**诊断：**

.. code:: bash

   ray status

**解决方案：**

.. code:: bash

   ray stop
   ray start --head

GPU 资源分配失败
~~~~~~~~~~~~~~~~

**现象：** ``ValueError: Not enough GPUs`` 或 ``actor_gpus + rollout_gpus > available``

**解决方案：** 确保 ``trainer.actor_gpus + trainer.rollout_gpus <= 总 GPU 数``：

.. code:: yaml

   trainer:
     actor_gpus: 4
     rollout_gpus: 4
     # 总和必须 <= N_GPUS_PER_NODE × NNODES

SGLang 引擎启动失败
~~~~~~~~~~~~~~~~~~~~

**现象：** RolloutManager 初始化时 ``TimeoutError``

**根因：** SGLang 引擎启动失败（模型加载、内存问题）。

**诊断：**

.. code:: bash

   grep -i "error\|oom\|cuda" siirl_logs/*.log

**解决方案：**

- 降低 ``rollout.gpu_memory_utilization``\ （默认 0.5）
- 确认模型路径正确且可访问
- 检查 CUDA 版本兼容性

训练循环问题
------------

训练 OOM
~~~~~~~~

**现象：** ``torch.cuda.OutOfMemoryError``

**按优先级解决：**

1. 减小 ``actor.ppo_micro_batch_size_per_gpu``
2. 启用 ``megatron.param_offload: true``
3. 减小 ``data.max_response_length``
4. 增加训练 GPU 数

Colocated 模式 OOM
~~~~~~~~~~~~~~~~~~

**现象：** rollout↔train 切换时 OOM

**解决方案：**

- 框架自动将 ``gpu_memory_utilization`` 限制在 0.45，**不要手动覆盖**
- 启用 ``rollout.colocate_release_weights_during_sync: true``
- 增大 ``trainer.colocate_timeout_s`` 以适应慢速卸载

训练停滞/无进展
~~~~~~~~~~~~~~~

**现象：** 长时间无训练步日志

**根因：** Rollout 未产出样本（工具超时、所有样本失败）。

**诊断：**

.. code:: bash

   grep "DataBuffer" siirl_logs/*.log | tail -20

**解决方案：**

- 检查工具环境是否可用（AIO Proxy 是否运行）
- 增大 ``rollout.train_server_concurrency``
- 检查 ``max_response_length`` 是否过小

奖励始终为零
~~~~~~~~~~~~

**现象：** 所有步骤 ``reward/mean = 0.0``

**解决方案：**

- 验证自定义奖励函数返回非零值
- 检查 ``data.reward_fn_key`` 与数据集列名匹配
- 多轮场景：确认奖励函数兼容工具交互轨迹

多轮 / Agentic 问题
--------------------

工具调用未被检测
~~~~~~~~~~~~~~~~

**现象：** 模型生成了工具调用语法但 rollout 未执行工具

**解决方案：**

.. code:: yaml

   rollout:
     multiturn:
       env_kwargs:
         tool_format: hermes    # 必须与模型的工具调用格式匹配

AIO Proxy 连接失败
~~~~~~~~~~~~~~~~~~

**现象：** ``Error: Exception occurred while getting server from master node``

**解决方案：**

.. code:: bash

   # 验证 AIO Proxy 是否运行
   curl http://PROXY_HOST:PROXY_PORT/health

   # 如果未运行则启动
   python -m aio.Scheduler.proxy --config aio_config.yaml

工具环境超时
~~~~~~~~~~~~

**解决方案：**

- 增加 AIO 配置中的工具实例数量
- 检查 AIO ResourcePool 容量：``curl http://proxy/status``
- 减小 ``max_parallel_calls``

轨迹过短
~~~~~~~~

**现象：** 大多数 rollout 在第 1 轮就终止

**解决方案：**

.. code:: yaml

   rollout:
     multiturn:
       max_env_turns: 5
       max_assistant_turns: 10

检查点问题
----------

恢复失败
~~~~~~~~

**现象：** 恢复时 ``FileNotFoundError``

**解决方案：**

.. code:: yaml

   trainer:
     resume_mode: auto              # 自动检测最新检查点
     # 或明确指定：
     resume_mode: resume_path
     resume_from_path: /path/to/checkpoint

检查点过大
~~~~~~~~~~

**解决方案：** 只保存必要内容：

.. code:: yaml

   actor_ref:
     checkpoint:
       save_contents: ["model"]      # 跳过优化器和额外状态

性能问题
--------

GPU 利用率低
~~~~~~~~~~~~

**现象：** ``nvidia-smi`` 显示 <50% GPU 使用率

**解决方案：**

- 增大 ``trainer.async_factor`` 以缓冲更多 rollout 批次
- 启用 off-policy：``trainer.off_policy_step: 2``
- 增加 rollout GPU 数
- 扩展 AIO 工具实例

参数同步慢
~~~~~~~~~~

**现象：** 训练步之间有明显停顿

**解决方案：**

- 增大 ``trainer.param_sync_buffer_size`` 以分块传输
- 使用 ``trainer.param_sync_rpc_timeout_s`` 检测挂起

获取帮助
--------

如果以上方案无法解决你的问题：

1. 检查 ``siirl_logs/`` 目录中的日志
2. 设置 ``LOGURU_LEVEL=DEBUG`` 获取详细日志
3. 在 https://github.com/sii-research/siirl-agentic/issues 提交 Issue
