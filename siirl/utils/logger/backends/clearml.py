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
ClearML Logger Backend

Provides integration with ClearML for experiment tracking.
"""

from typing import Any

from loguru import logger

from .base import BackendConfig, NumericScalar


class ClearMLBackend:
    """
    ClearML logger backend.

    Features:
        - Lazy initialization
        - Automatic task continuation
        - Table logging support

    Configuration via BackendConfig.extra:
        - output_uri: Output URI for artifacts (default: False)
        - continue_last_task: Continue last task if exists (default: True)
    """

    def __init__(self, config: BackendConfig):
        """
        Initialize ClearML backend.

        Args:
            config: Backend configuration
        """
        self._name = "clearml"
        self._config = config
        self._initialized = False
        self._task = None

    @property
    def name(self) -> str:
        return self._name

    def _ensure_initialized(self) -> bool:
        """
        Ensure ClearML is initialized.

        Returns:
            True if successfully initialized, False otherwise
        """
        if self._initialized:
            return True

        try:
            import clearml

            extra = self._config.extra or {}
            output_uri = extra.get("output_uri", False)
            continue_last_task = extra.get("continue_last_task", True)

            # Initialize ClearML task
            self._task = clearml.Task.init(
                task_name=self._config.experiment_name,
                project_name=self._config.project_name,
                continue_last_task=continue_last_task,
                output_uri=output_uri,
            )

            # Connect configuration
            if self._config.config:
                self._task.connect_configuration(self._config.config, name="Hyperparameters")

            self._initialized = True
            logger.success(f"ClearML initialized: project={self._config.project_name}, task={self._config.experiment_name}")
            return True

        except ImportError:
            logger.warning("clearml not installed. Install with: pip install clearml")
            return False
        except Exception as e:
            logger.error(f"ClearML initialization failed: {e}")
            return False

    def _get_logger(self):
        """Get ClearML logger instance."""
        return self._task.get_logger() if self._task else None

    def log(self, data: dict[str, NumericScalar], step: int) -> None:
        """
        Log scalar metrics to ClearML.

        Args:
            data: Dictionary of metric names to values
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            import numpy as np

            clearml_logger = self._get_logger()
            if not clearml_logger:
                return

            for k, v in data.items():
                # Parse title/series from key (e.g., "actor/loss" -> title="actor", series="loss")
                if "/" in k:
                    title, series = k.split("/", 1)
                else:
                    title, series = "metrics", k

                if isinstance(v, int | float | np.floating | np.integer):
                    clearml_logger.report_scalar(
                        title=title,
                        series=series,
                        value=float(v),
                        iteration=step,
                    )
        except Exception as e:
            logger.error(f"ClearML log error: {e}")

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log text content to ClearML.

        Args:
            tag: Tag for the text
            text: Text content
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            clearml_logger = self._get_logger()
            if not clearml_logger:
                return

            if "/" in tag:
                title, series = tag.split("/", 1)
            else:
                title, series = "text", tag

            # Prevent unused variable warning
            _ = title
            _ = series

            clearml_logger.report_text(text, level=0, print_console=False)
        except Exception as e:
            logger.error(f"ClearML log_text error: {e}")

    def log_table(self, tag: str, columns: list[str], data: list[list[Any]], step: int) -> None:
        """
        Log tabular data to ClearML.

        Args:
            tag: Tag for the table
            columns: Column names
            data: Rows of data
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            import pandas as pd

            clearml_logger = self._get_logger()
            if not clearml_logger:
                return

            if "/" in tag:
                title, series = tag.split("/", 1)
            else:
                title, series = "tables", tag

            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)

            clearml_logger.report_table(
                title=title,
                series=series,
                table_plot=df,
                iteration=step,
            )
        except Exception as e:
            logger.error(f"ClearML log_table error: {e}")

    def finish(self) -> None:
        """
        Mark the ClearML task as completed.
        """
        if self._initialized and self._task:
            try:
                self._task.mark_completed()
                logger.info("ClearML task completed")
            except Exception as e:
                logger.error(f"ClearML finish error: {e}")

        self._initialized = False
        self._task = None
