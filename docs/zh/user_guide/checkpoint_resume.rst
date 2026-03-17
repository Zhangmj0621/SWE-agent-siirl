检查点与恢复
============

   **适合谁：** 需要管理检查点和恢复训练的用户。

   **你将获得：** 检查点配置、恢复模式说明与 HuggingFace 模型导出方法。

保存配置
--------

.. code:: yaml

   trainer:
     save_freq: 10                    # 每 N 步保存一次（-1 = 禁用）
     max_actor_ckpt_to_keep: 5       # 保留最近 N 个 actor 检查点
     max_critic_ckpt_to_keep: 5      # 保留最近 N 个 critic 检查点
     default_local_dir: checkpoints/  # 检查点目录

   actor_ref:
     checkpoint:
       save_contents: ["model", "optimizer", "extra"]  # 保存内容
       # 添加 "hf_model" 同时导出 HuggingFace 格式

恢复训练
--------

.. code:: yaml

   trainer:
     resume_mode: auto           # 自动检测最新检查点
     # 或者指定路径：
     resume_mode: resume_path
     resume_from_path: /path/to/checkpoint/step_100

恢复模式说明
~~~~~~~~~~~~

=============== ===============================================
模式             行为
=============== ===============================================
``auto``        在 ``default_local_dir`` 中找最新检查点
``disable``     从头开始，忽略已有检查点
``resume_path`` 从指定 ``resume_from_path`` 恢复
=============== ===============================================

检查点内容
----------

============= ==================================== ========
内容          说明                                 默认
============= ==================================== ========
``model``     模型权重（Megatron 格式）             已保存
``optimizer`` 优化器状态                           已保存
``extra``     RNG 状态、lr_scheduler、step 计数    已保存
``hf_model``  HuggingFace 格式（转换后）           不保存
============= ==================================== ========

异步检查点保存
--------------

.. code:: yaml

   actor_ref:
     checkpoint:
       async_save: true    # 实验性：非阻塞检查点保存
