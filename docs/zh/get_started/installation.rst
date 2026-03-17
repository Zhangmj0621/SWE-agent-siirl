安装指南
========

   **适合谁：** 首次安装 siirl-agentic 的用户。

   **你将获得：** 在本地系统上验证通过的 siirl-agentic 安装环境。

环境要求
--------

- Python >= 3.10
- CUDA >= 12.4（GPU 支持）
- PyTorch >= 2.8.0
- Ray >= 2.53.0

已验证环境
~~~~~~~~~~

=============== ===========
组件            版本
=============== ===========
Python          3.10+
PyTorch         2.8.0
CUDA            12.8
Flash Attention 2.8.2
Transformers    4.57.0
Ray             2.53.0
SGLang          0.5.5.post3
Triton          3.4.0
=============== ===========

快速安装
--------

.. code:: bash

   pip install siirl-agentic

开发环境安装
------------

.. code:: bash

   git clone https://github.com/sii-research/siirl-agentic.git
   cd siirl-agentic
   pip install -e ".[dev]"

GPU 安装（Flash Attention + Triton）
------------------------------------

.. code:: bash

   pip install "siirl-agentic[gpu]"

完整安装（所有可选依赖）
------------------------

.. code:: bash

   pip install "siirl-agentic[all]"

从源码安装（固定版本）
----------------------

.. code:: bash

   git clone https://github.com/sii-research/siirl-agentic.git
   cd siirl-agentic

   # 先安装 PyTorch（CUDA 12.4+）
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

   # 安装 Flash Attention
   pip install flash-attn==2.8.2 --no-build-isolation

   # 安装本包
   pip install -e ".[dev]"

验证安装
--------

.. code:: bash

   # 检查版本
   python -c "import siirl; print(siirl.__version__)"

**预期输出：**

::

   0.1.0

.. code:: bash

   # 检查 GPU 支持
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, version: {torch.version.cuda}')"

**预期输出：**

::

   CUDA: True, version: 12.8

环境变量
--------

.. code:: bash

   # 日志
   export LOGURU_LEVEL=INFO
   export SIIRL_LOG_DIRECTORY=siirl_logs

   # Ray
   export RAY_memory_monitor_refresh_ms=0

   # 分布式训练
   export MASTER_ADDR=localhost
   export MASTER_PORT=29500

下一步
------

- :doc:`快速开始 <quickstart>` — 运行首个 GRPO 训练任务
- :doc:`首个 Agentic 训练任务 <first_agentic_training_job>` — 带工具交互的训练
