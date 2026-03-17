Troubleshooting
===============

   **Who this is for:** Users encountering issues during training.

   **What you will get:** Solutions to the most common failure modes.

Initialization Failures
-----------------------

Ray Connection Issues
~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``ConnectionError: Ray is not initialized``

**Root cause:** Ray cluster not started or conflicting Ray processes.

**Diagnosis:**

.. code:: bash

   ray status

**Fix:**

.. code:: bash

   ray stop
   ray start --head

GPU Resource Allocation Failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``ValueError: Not enough GPUs`` or ``actor_gpus + rollout_gpus > available``

**Root cause:** Requested GPU count exceeds available resources.

**Fix:** Ensure ``trainer.actor_gpus + trainer.rollout_gpus <= total GPUs``:

.. code:: yaml

   trainer:
     actor_gpus: 4
     rollout_gpus: 4
     # Total must be <= N_GPUS_PER_NODE × NNODES

SGLang Engine Startup Failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``TimeoutError`` during RolloutManager initialization

**Root cause:** SGLang engine failed to start (model loading, memory).

**Diagnosis:**

.. code:: bash

   # Check SGLang logs
   grep -i "error\|oom\|cuda" siirl_logs/*.log

**Fix:** - Reduce ``rollout.gpu_memory_utilization`` (default: 0.5) - Ensure model path is correct and accessible - Check CUDA version compatibility

Training Loop Issues
--------------------

OOM During Training
~~~~~~~~~~~~~~~~~~~

**Symptom:** ``torch.cuda.OutOfMemoryError``

**Root cause:** Batch too large for available GPU memory.

**Fix (in order of priority):** 1. Reduce ``actor.ppo_micro_batch_size_per_gpu`` 2. Enable ``megatron.param_offload: true`` 3. Reduce ``data.max_response_length`` 4. Add more training GPUs

OOM in Colocated Mode
~~~~~~~~~~~~~~~~~~~~~

**Symptom:** OOM during rollout↔train transition

**Root cause:** Insufficient memory after weight offloading.

**Fix:** - Framework auto-clamps ``gpu_memory_utilization`` to 0.45 — do not override - Enable ``rollout.colocate_release_weights_during_sync: true`` - Increase ``trainer.colocate_timeout_s`` for slow offload

Training Stuck / No Progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** No training step logs for extended period

**Root cause:** Rollout producing no samples (tool timeouts, all samples failing).

**Diagnosis:**

.. code:: bash

   # Check DataBuffer status
   grep "DataBuffer" siirl_logs/*.log | tail -20

**Fix:** - Check tool environment availability (AIO Proxy running?) - Increase ``rollout.train_server_concurrency`` - Check ``max_response_length`` isn’t too small

Reward Always Zero
~~~~~~~~~~~~~~~~~~

**Symptom:** ``reward/mean = 0.0`` across all steps

**Root cause:** Reward function not matching expected format.

**Fix:** - Verify custom reward function returns non-zero values - Check ``data.reward_fn_key`` matches dataset column - For multi-turn: ensure reward function handles tool interaction trajectories

Multi-Turn / Agentic Issues
---------------------------

Tool Calls Not Detected
~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Model generates tool call syntax but rollout doesn’t execute tools

**Root cause:** Tool parser format mismatch.

**Fix:**

.. code:: yaml

   rollout:
     multiturn:
       env_kwargs:
         tool_format: hermes    # Must match model's tool call format

AIO Proxy Connection Failed
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``Error: Exception occurred while getting server from master node``

**Root cause:** AIO Proxy not running or unreachable.

**Fix:**

.. code:: bash

   # Verify AIO Proxy is running
   curl http://PROXY_HOST:PROXY_PORT/health

   # Start if not running
   python -m aio.Scheduler.proxy --config aio_config.yaml

Tool Environment Timeout
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``TimeoutError`` during tool execution

**Root cause:** Tool instance overloaded or unresponsive.

**Fix:** - Increase tool instance count in AIO config - Check AIO ResourcePool capacity: ``curl http://proxy/status`` - Reduce ``max_parallel_calls`` per sample

Trajectories Too Short
~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Most rollouts terminate after 1 turn

**Root cause:** ``max_assistant_turns=1`` or ``max_env_turns=1``

**Fix:**

.. code:: yaml

   rollout:
     multiturn:
       max_env_turns: 5
       max_assistant_turns: 10

Checkpoint Issues
-----------------

Resume Fails
~~~~~~~~~~~~

**Symptom:** ``FileNotFoundError`` on resume

**Root cause:** Checkpoint path doesn’t exist or incompatible format.

**Fix:**

.. code:: yaml

   trainer:
     resume_mode: auto              # Auto-detect latest checkpoint
     # Or specify explicitly:
     resume_mode: resume_path
     resume_from_path: /path/to/checkpoint

Checkpoint Too Large
~~~~~~~~~~~~~~~~~~~~

**Symptom:** Slow save/load, disk full

**Fix:** Save only essential contents:

.. code:: yaml

   actor_ref:
     checkpoint:
       save_contents: ["model"]      # Skip optimizer and extra states

Performance Issues
------------------

Low GPU Utilization
~~~~~~~~~~~~~~~~~~~

**Symptom:** ``nvidia-smi`` shows <50% GPU usage

**Root cause:** Rollout bottleneck (slow tools) or data pipeline stall.

**Fix:** - Increase ``trainer.async_factor`` to buffer more rollout batches - Enable off-policy: ``trainer.off_policy_step: 2`` - Add more rollout GPUs - Scale AIO tool instances

Slow Parameter Sync
~~~~~~~~~~~~~~~~~~~

**Symptom:** Long pause between training steps

**Root cause:** Large model weights, slow network.

**Fix:** - Increase ``trainer.param_sync_buffer_size`` for chunked transfer - Use ``trainer.param_sync_rpc_timeout_s`` to detect hangs

Getting Help
------------

If the above doesn’t resolve your issue:

1. Check logs in ``siirl_logs/`` directory
2. Set ``LOGURU_LEVEL=DEBUG`` for verbose logging
3. File an issue at https://github.com/sii-research/siirl-agentic/issues
