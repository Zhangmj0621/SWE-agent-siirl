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
MetricTracker - Unified interface for logging to multiple backends.

Provides a single entry point for logging metrics, text, and tables to
multiple backends (console, wandb, tensorboard) simultaneously.
"""

import contextlib
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .backends import BackendConfig, BackendRegistry
from .backends.base import LoggerBackend


@dataclass
class GenerationSample:
    """
    Represents a single generation sample for validation logging.

    Attributes:
        input_text: The input prompt
        output_text: The generated output
        score: The reward/score for this generation
        metadata: Optional additional metadata
    """

    input_text: str
    output_text: str
    score: float
    metadata: dict[str, Any] | None = None


class MetricTracker:
    """
    Unified metric tracker that manages multiple logger backends.

    This class provides a single interface for logging to multiple backends
    (console, wandb, tensorboard) simultaneously. It handles initialization,
    logging, and cleanup for all configured backends.

    Usage:
        # Basic usage (eager initialization - validates connections immediately)
        tracker = MetricTracker(
            project_name="siirl_agentic",
            experiment_name="gsm8k_grpo",
            backends=["console", "wandb"],
            config={"lr": 1e-4, "batch_size": 32}
        )

        tracker.log({"loss": 0.5, "accuracy": 0.9}, step=100)
        tracker.finish()

        # With context manager
        with MetricTracker("project", "exp", ["console"]) as tracker:
            tracker.log({"loss": 0.5}, step=1)

    Args:
        project_name: Project name for experiment organization
        experiment_name: Name of this specific experiment
        backends: List of backend names to use, or single backend name
        config: Training configuration to log to backends that support it
        backend_configs: Per-backend configuration, e.g., {"wandb": {"proxy": "..."}}
        eager_init: If True (default), validate all backend connections immediately.
                    If False, defer connection until first log() call (lazy loading).
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        backends: str | list[str] = "console",
        config: dict[str, Any] | None = None,
        backend_configs: dict[str, dict[str, Any]] | None = None,
        eager_init: bool = True,
    ):
        self._project_name = project_name
        self._experiment_name = experiment_name
        self._config = config
        self._backend_configs = backend_configs or {}
        self._backends: dict[str, LoggerBackend] = {}
        self._closed = False
        self._eager_init = eager_init

        # Normalize backends to list
        if isinstance(backends, str):
            backends = [backends]

        # Initialize each backend
        for backend_name in backends:
            self._init_backend(backend_name)

        active_backends = list(self._backends.keys())
        logger.info(f"MetricTracker initialized: project={project_name}, " f"experiment={experiment_name}, backends={active_backends}")

        # Eager initialization: validate all connections immediately
        if eager_init and active_backends:
            self._validate_backends()

    def _init_backend(self, name: str) -> None:
        """
        Initialize a single backend.

        Args:
            name: Backend name to initialize
        """
        try:
            backend_config = BackendConfig(
                project_name=self._project_name,
                experiment_name=self._experiment_name,
                config=self._config,
                extra=self._backend_configs.get(name, {}),
            )

            backend = BackendRegistry.create(name, backend_config)
            self._backends[name] = backend

        except ValueError as e:
            logger.warning(f"Backend '{name}' not found: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize backend '{name}': {e}")

    def _validate_backends(self) -> None:
        """
        Eagerly validate all backend connections.

        This method is called during __init__ when eager_init=True.
        It triggers the lazy initialization in each backend to catch
        configuration errors early (e.g., invalid wandb API key, network issues).

        Raises:
            RuntimeError: If any critical backend fails to initialize
        """
        failed_backends = []

        for name, backend in list(self._backends.items()):
            try:
                # All backends implement _ensure_initialized()
                success = backend._ensure_initialized()
                if not success:
                    failed_backends.append((name, "initialization returned False"))
                    self._backends.pop(name, None)

            except Exception as e:
                failed_backends.append((name, str(e)))
                self._backends.pop(name, None)

        # Report validation results
        if failed_backends:
            for name, error in failed_backends:
                logger.error(f"Backend '{name}' validation failed: {error}")

            if not self._backends:
                raise RuntimeError(f"All logging backends failed to initialize: {failed_backends}")
            else:
                logger.warning(f"Some backends failed, continuing with: {list(self._backends.keys())}")
        else:
            logger.success(f"All backends validated successfully: {list(self._backends.keys())}")

    def add_backend(self, name: str, extra_config: dict[str, Any] | None = None) -> bool:
        """
        Dynamically add a new backend.

        Args:
            name: Backend name to add
            extra_config: Additional configuration for the backend

        Returns:
            True if successfully added, False otherwise
        """
        if name in self._backends:
            logger.warning(f"Backend '{name}' already exists")
            return False

        if extra_config:
            self._backend_configs[name] = extra_config

        self._init_backend(name)
        return name in self._backends

    def remove_backend(self, name: str) -> bool:
        """
        Remove a backend.

        Args:
            name: Backend name to remove

        Returns:
            True if successfully removed, False otherwise
        """
        if name not in self._backends:
            return False

        backend = self._backends.pop(name)
        try:
            backend.finish()
        except Exception as e:
            logger.warning(f"Error finishing backend '{name}': {e}")

        return True

    def log(
        self,
        data: dict[str, float],
        step: int,
        backends: list[str] | None = None,
    ) -> None:
        """
        Log scalar metrics to all (or specified) backends.

        Args:
            data: Dictionary of metric names to values
            step: Current training step
            backends: Optional list of backend names to log to (None = all)
        """
        if self._closed:
            logger.warning("MetricTracker is closed, ignoring log call")
            return

        for name, backend in self._backends.items():
            if backends is None or name in backends:
                try:
                    backend.log(data, step)
                except Exception as e:
                    logger.error(f"Backend '{name}' log error: {e}")

    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
        backends: list[str] | None = None,
    ) -> None:
        """
        Log text content to all (or specified) backends.

        Args:
            tag: Tag/label for the text
            text: Text content to log
            step: Current training step
            backends: Optional list of backend names to log to (None = all)
        """
        if self._closed:
            return

        for name, backend in self._backends.items():
            if backends is None or name in backends:
                try:
                    backend.log_text(tag, text, step)
                except Exception as e:
                    logger.error(f"Backend '{name}' log_text error: {e}")

    def log_generation(
        self,
        samples: list[GenerationSample],
        step: int,
        tag: str = "val/generations",
        backends: list[str] | None = None,
    ) -> None:
        """
        Log generation samples (for validation).

        Converts GenerationSample objects to table format for logging.

        Args:
            samples: List of generation samples
            step: Current training step
            tag: Tag for the table (default: "val/generations")
            backends: Optional list of backend names to log to (None = all)
        """
        if self._closed or not samples:
            return

        # Prepare table data
        columns = ["input", "output", "score"]
        data = [
            [
                s.input_text[:200] if len(s.input_text) > 200 else s.input_text,
                s.output_text[:200] if len(s.output_text) > 200 else s.output_text,
                f"{s.score:.4f}" if isinstance(s.score, float) else str(s.score),
            ]
            for s in samples
        ]

        for name, backend in self._backends.items():
            if backends is None or name in backends:
                try:
                    backend.log_table(tag, columns, data, step)
                except Exception as e:
                    logger.error(f"Backend '{name}' log_generation error: {e}")

    def finish(self) -> None:
        """
        Close all backends and release resources.

        Should be called when logging is complete. After calling finish(),
        subsequent log calls will be ignored.
        """
        if self._closed:
            return

        for name, backend in self._backends.items():
            try:
                backend.finish()
            except Exception as e:
                logger.error(f"Backend '{name}' finish error: {e}")

        self._backends.clear()
        self._closed = True
        logger.info("MetricTracker closed")

    @property
    def active_backends(self) -> list[str]:
        """Get list of currently active backend names."""
        return list(self._backends.keys())

    @property
    def is_closed(self) -> bool:
        """Check if tracker has been closed."""
        return self._closed

    def __enter__(self) -> "MetricTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures finish() is called."""
        self.finish()

    def __del__(self):
        """Destructor - last resort cleanup."""
        if not self._closed:
            with contextlib.suppress(Exception):
                self.finish()
