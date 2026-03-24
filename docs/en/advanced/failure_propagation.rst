Failure Propagation & Graceful Shutdown
=======================================

   **Who this is for:** Users running large-scale distributed training who need to understand failure handling behavior.

   **What you will get:** Understanding of how failures are detected, propagated, and handled across distributed components.

Overview
--------

In a distributed MPMD system, a failure in one component (e.g., CUDA OOM in the Trainer) must be propagated to all other components to prevent zombie processes and wasted resources. siirl-agentic uses a centralized ``TaskCoordinator`` Ray actor for lifecycle management.

TaskCoordinator
---------------

The ``TaskCoordinator`` (``siirl/utils/task_coordinator.py``) is a Ray actor that provides:

- **Unified stop signal** for all components
- **Failure propagation** across Trainer, RolloutManager, and DataCoordinator
- **Graceful shutdown** coordination
- **Event logging** for post-mortem debugging

Lifecycle States
~~~~~~~~~~~~~~~~

::

   RUNNING  ──► COMPLETED   (normal finish)
      │
      ├──────► FAILED       (error in any component)
      │
      └──────► SHUTDOWN     (graceful shutdown requested)

All non-RUNNING states cause ``should_stop()`` to return ``True``.

How Components Use the Coordinator
----------------------------------

1. Initialization
~~~~~~~~~~~~~~~~~

The coordinator is created in ``MainRunner`` and passed to all components:

.. code:: python

   coordinator = TaskCoordinator.remote()
   trainer = Trainer(..., coordinator=coordinator)
   rollout_manager = RolloutManager(..., coordinator=coordinator)

2. Polling for Stop Signal
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each component checks ``should_stop()`` periodically in its main loop:

.. code:: python

   while True:
       if ray.get(self.coordinator.should_stop.remote()):
           logger.info("Stop signal received, exiting...")
           break
       # ... continue work ...

3. Reporting Failures
~~~~~~~~~~~~~~~~~~~~~

When a component encounters an error, it reports to the coordinator:

.. code:: python

   try:
       self.train_step(batch)
   except Exception as e:
       ray.get(self.coordinator.report_failure.remote(
           source="trainer_0",
           reason=str(e)
       ))
       raise

4. Requesting Shutdown
~~~~~~~~~~~~~~~~~~~~~~

For normal completion (e.g., all epochs finished):

.. code:: python

   ray.get(self.coordinator.request_shutdown.remote(
       source="main_runner",
       reason="Training completed"
   ))

Failure Scenarios
-----------------

CUDA OOM in Trainer
~~~~~~~~~~~~~~~~~~~

1. Trainer catches ``RuntimeError: CUDA out of memory``
2. Calls ``coordinator.report_failure.remote("trainer", "CUDA OOM")``
3. Coordinator sets status to ``FAILED``
4. RolloutManager’s next ``should_stop()`` call returns ``True``
5. RolloutManager cancels pending rollouts and exits
6. ``MainRunner`` detects failure via ``coordinator.get_summary()`` and runs cleanup

Rollout Timeout
~~~~~~~~~~~~~~~

1. RolloutManager detects rollout exceeding max time
2. Reports failure with timeout details
3. Trainer stops consuming from DataBuffer
4. All components shut down gracefully

Ray Actor Crash
~~~~~~~~~~~~~~~

If a Ray actor crashes without reporting:

1. ``MainRunner`` monitors actor health via Ray’s built-in actor failure detection
2. On detecting dead actor, calls ``coordinator.report_failure.remote()``
3. Remaining actors receive stop signal on next poll

Debugging Failures
------------------

Event Log
~~~~~~~~~

Retrieve the full event history for post-mortem analysis:

.. code:: python

   events = ray.get(coordinator.get_events.remote())
   for event in events:
       print(f"[{event['timestamp']}] {event['source']}: {event['message']}")

Summary
~~~~~~~

Get a quick overview of the coordination state:

.. code:: python

   summary = ray.get(coordinator.get_summary.remote())
   # {
   #     "status": "failed",
   #     "failure_reason": "CUDA OOM in trainer_0",
   #     "duration_seconds": 3421.5,
   #     "event_count": 47
   # }

Best Practices
--------------

1. **Poll frequency:** Check ``should_stop()`` at least once per training step or rollout batch — not inside tight inner loops.
2. **Always report:** Catch exceptions at the component boundary and report before re-raising.
3. **Cleanup resources:** Use try/finally blocks to ensure GPU memory, Ray object refs, and temp files are cleaned up.
4. **Log context:** Include component name, GPU rank, and error details in failure reports.

Related
-------

- :doc:`Architecture Overview <../concepts/architecture_overview>` — MPMD component roles
- :doc:`Performance Tuning <performance_tuning>` — Preventing OOM via offloading and batching
- :doc:`Troubleshooting <../faq/troubleshooting>` — Common failure patterns and fixes
