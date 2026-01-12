# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Task Coordinator for distributed training lifecycle management.

This module provides a centralized coordination mechanism for managing the lifecycle
of distributed training tasks, including graceful shutdown and failure handling.

Usage:
    # Create coordinator in main orchestrator
    coordinator = TaskCoordinator.remote()

    # Pass to components
    trainer = Trainer(..., coordinator=coordinator)

    # In training loop, check if should stop
    if ray.get(coordinator.should_stop.remote()):
        break

    # Report failures
    ray.get(coordinator.report_failure.remote("trainer_0", "CUDA OOM"))
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ray


class TaskStatus(Enum):
    """Task lifecycle status."""

    RUNNING = "running"  # Task is running normally
    COMPLETED = "completed"  # Task completed successfully
    FAILED = "failed"  # Task failed with error
    SHUTDOWN = "shutdown"  # Graceful shutdown requested


@dataclass
class TaskEvent:
    """Record of a task lifecycle event."""

    timestamp: float
    source: str
    event_type: str  # "failure", "shutdown", "completed"
    message: str


@ray.remote
class TaskCoordinator:
    """
    Centralized coordinator for distributed training task lifecycle.

    This Ray actor provides:
    - Unified stop signal for all components
    - Failure propagation across components
    - Graceful shutdown coordination
    - Event logging for debugging

    All components should:
    1. Receive coordinator handle during initialization
    2. Periodically call should_stop() in their main loops
    3. Call report_failure() when encountering errors
    4. Call request_shutdown() when task should end normally

    Example:
        # In Trainer.train()
        while True:
            if ray.get(self.coordinator.should_stop.remote()):
                logger.info("Stop signal received, exiting...")
                break

            try:
                self.train_step(batch)
            except Exception as e:
                ray.get(self.coordinator.report_failure.remote(
                    source="trainer_0",
                    reason=str(e)
                ))
                raise
    """

    def __init__(self):
        """Initialize coordinator with RUNNING status."""
        self._status = TaskStatus.RUNNING
        self._failure_reason: str | None = None
        self._events: list = []
        self._start_time = time.time()

        self._log_event("coordinator", "started", "TaskCoordinator initialized")

    # =========================================================================
    # Query Methods - Called by components to check status
    # =========================================================================

    def should_stop(self) -> bool:
        """
        Check if components should stop their execution.

        Components should call this periodically in their main loops.

        Returns:
            True if task should stop (completed, failed, or shutdown requested)
            False if task should continue running

        Example:
            while True:
                if ray.get(coordinator.should_stop.remote()):
                    break
                # ... continue work ...
        """
        return self._status != TaskStatus.RUNNING

    def get_status(self) -> str:
        """
        Get current task status as string.

        Returns:
            One of: "running", "completed", "failed", "shutdown"
        """
        return self._status.value

    def get_failure_reason(self) -> str | None:
        """
        Get the reason for failure or shutdown.

        Returns:
            Failure/shutdown reason string, or None if still running
        """
        return self._failure_reason

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the task coordination state.

        Useful for debugging and logging.

        Returns:
            Dictionary with status, duration, failure reason, and event count
        """
        return {
            "status": self._status.value,
            "failure_reason": self._failure_reason,
            "duration_seconds": time.time() - self._start_time,
            "event_count": len(self._events),
        }

    def get_events(self) -> list:
        """
        Get all recorded events for debugging.

        Returns:
            List of TaskEvent dictionaries
        """
        return [
            {
                "timestamp": e.timestamp,
                "source": e.source,
                "event_type": e.event_type,
                "message": e.message,
            }
            for e in self._events
        ]

    # =========================================================================
    # State Update Methods - Called by components to report status changes
    # =========================================================================

    def report_failure(self, source: str, reason: str) -> bool:
        """
        Report a failure from a component.

        When any component fails, it should call this method to notify
        other components to stop gracefully.

        Only the first failure is recorded; subsequent failures are logged
        but don't change the failure reason.

        Args:
            source: Identifier of the failing component (e.g., "trainer_0", "rollout")
            reason: Human-readable failure reason

        Returns:
            True if this was the first failure reported, False otherwise

        Example:
            try:
                self.train_step(batch)
            except Exception as e:
                ray.get(coordinator.report_failure.remote("trainer_0", str(e)))
                raise
        """
        self._log_event(source, "failure", reason)

        if self._status == TaskStatus.RUNNING:
            self._status = TaskStatus.FAILED
            self._failure_reason = f"[{source}] {reason}"
            return True
        return False

    def report_completed(self, source: str = "unknown") -> bool:
        """
        Report successful task completion.

        Called when training finishes normally (e.g., reached max epochs).

        Args:
            source: Identifier of the component reporting completion

        Returns:
            True if status changed to completed, False if already stopped

        Example:
            if current_epoch >= max_epochs:
                ray.get(coordinator.report_completed.remote("main_runner"))
        """
        self._log_event(source, "completed", "Task completed successfully")

        if self._status == TaskStatus.RUNNING:
            self._status = TaskStatus.COMPLETED
            self._failure_reason = f"Completed by {source}"
            return True
        return False

    def request_shutdown(self, reason: str, source: str = "unknown") -> bool:
        """
        Request graceful shutdown of all components.

        Called when task should end normally but before natural completion,
        e.g., when training data is exhausted.

        Args:
            reason: Reason for shutdown request
            source: Identifier of the component requesting shutdown

        Returns:
            True if shutdown was initiated, False if already stopped

        Example:
            if batch_data is None:
                ray.get(coordinator.request_shutdown.remote(
                    reason="Training data exhausted",
                    source="trainer_0"
                ))
        """
        self._log_event(source, "shutdown", reason)

        if self._status == TaskStatus.RUNNING:
            self._status = TaskStatus.SHUTDOWN
            self._failure_reason = f"[{source}] {reason}"
            return True
        return False

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _log_event(self, source: str, event_type: str, message: str):
        """Record an event for debugging."""
        event = TaskEvent(
            timestamp=time.time(),
            source=source,
            event_type=event_type,
            message=message,
        )
        self._events.append(event)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_coordinator() -> "ray.actor.ActorHandle":
    """
    Create a TaskCoordinator actor.

    Returns:
        Ray actor handle to TaskCoordinator

    Example:
        coordinator = create_coordinator()
        trainer = Trainer(..., coordinator=coordinator)
    """
    return TaskCoordinator.remote()
