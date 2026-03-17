Installation
============

   **Who this is for:** New users setting up siirl-agentic for the first time.

   **What you will get:** A working siirl-agentic installation verified on your system.

Requirements
------------

- Python >= 3.10
- CUDA >= 12.4 (for GPU support)
- PyTorch >= 2.8.0
- Ray >= 2.53.0

Tested Environment
~~~~~~~~~~~~~~~~~~

=============== ===========
Package         Version
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

Quick Install
-------------

.. code:: bash

   pip install siirl-agentic

Development Install
-------------------

.. code:: bash

   git clone https://github.com/sii-research/siirl-agentic.git
   cd siirl-agentic
   pip install -e ".[dev]"

GPU Install (Flash Attention + Triton)
--------------------------------------

.. code:: bash

   pip install "siirl-agentic[gpu]"

Full Install (All Optional Dependencies)
----------------------------------------

.. code:: bash

   pip install "siirl-agentic[all]"

From Source with Pinned Versions
--------------------------------

.. code:: bash

   git clone https://github.com/sii-research/siirl-agentic.git
   cd siirl-agentic

   # Install PyTorch first (CUDA 12.4+)
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

   # Install Flash Attention
   pip install flash-attn==2.8.2 --no-build-isolation

   # Install the package
   pip install -e ".[dev]"

Verify Installation
-------------------

.. code:: bash

   # Check package version
   python -c "import siirl; print(siirl.__version__)"

**Expected output:**

::

   0.1.0

.. code:: bash

   # Check GPU support
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, version: {torch.version.cuda}')"

**Expected output:**

::

   CUDA: True, version: 12.8

Environment Variables
---------------------

.. code:: bash

   # Logging
   export LOGURU_LEVEL=INFO
   export SIIRL_LOG_DIRECTORY=siirl_logs

   # Ray
   export RAY_memory_monitor_refresh_ms=0

   # Distributed Training
   export MASTER_ADDR=localhost
   export MASTER_PORT=29500

Next Steps
----------

- :doc:`Quickstart <quickstart>` — Run your first GRPO training
- :doc:`First Agentic Training Job <first_agentic_training_job>` — Train with tool interaction
