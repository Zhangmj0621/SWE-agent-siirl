快速开始
========

   **适合谁：** 首次使用 siirl-agentic 的用户。

   **你将获得：** 5 分钟内运行第一个训练任务。

前提条件
--------

- Python >= 3.10
- NVIDIA GPU（A100 或更高推荐）
- CUDA 12.x
- Ray >= 2.53.0

1. 安装
-------

.. code:: bash

   # 克隆仓库
   git clone https://github.com/sii-research/siirl-agentic.git
   cd siirl-agentic

   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate

   # 安装
   pip install -e .

   # GPU 加速（可选）
   pip install flash-attn --no-build-isolation

2. 准备数据
-----------

创建训练数据文件（JSONL 格式）：

.. code:: json

   {"prompt": "Solve this coding task: ...", "reference": "expected output"}

3. 配置
-------

创建 YAML 配置文件 ``config.yaml``\ ：

.. code:: yaml

   data:
     train_files: "path/to/train.jsonl"
     max_prompt_length: 2048
     max_response_length: 4096

   actor_ref:
     actor:
       model_path: "Qwen/Qwen2.5-7B-Instruct"

   rollout:
     model_path: "Qwen/Qwen2.5-7B-Instruct"
     multiturn:
       env_type: tool_env
       max_env_turns: 5

   trainer:
     algorithm: grpo
     total_epochs: 1
     actor_gpus: 4
     rollout_gpus: 4

4. 启动训练
-----------

.. code:: bash

   python -m siirl.async_train --config config.yaml

5. 监控
-------

.. code:: bash

   # WandB（如已配置）
   # 打开 https://wandb.ai 查看训练曲线

   # TensorBoard
   tensorboard --logdir ./logs

下一步
------

- `安装指南 <installation.html>`__ — 完整安装说明
- `首个 Agentic 训练任务 <first_agentic_training_job.html>`__ — 详细的端到端教程
- `配置系统 <../user_guide/configuration_system.html>`__ — 深入理解配置层次

..

   完整英文版请参阅 `Quickstart (English) <../../en/get_started/quickstart.html>`__
