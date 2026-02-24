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
SwanLab Logger Backend

Provides integration with SwanLab for experiment tracking.
"""

import os
from typing import Any

from loguru import logger

from .base import BackendConfig, NumericScalar


class SwanLabBackend:
    """
    SwanLab logger backend.

    Features:
        - Lazy initialization
        - Cloud and local mode support
        - Text logging for generations

    Configuration via BackendConfig.extra:
        - api_key: SwanLab API key (or use SWANLAB_API_KEY env var)
        - log_dir: Local log directory (default: "swanlog")
        - mode: "cloud" or "local" (default: "cloud")

    Environment variables:
        - SWANLAB_API_KEY: SwanLab API key
        - SWANLAB_LOG_DIR: Local log directory
        - SWANLAB_MODE: "cloud" or "local"
    """

    def __init__(self, config: BackendConfig):
        """
        Initialize SwanLab backend.

        Args:
            config: Backend configuration
        """
        self._name = "swanlab"
        self._config = config
        self._initialized = False
        self._swanlab = None

    @property
    def name(self) -> str:
        return self._name

    def _ensure_initialized(self) -> bool:
        """
        Ensure SwanLab is initialized.

        Returns:
            True if successfully initialized, False otherwise
        """
        if self._initialized:
            return True

        try:
            import swanlab

            self._swanlab = swanlab

            # Get configuration from extra or environment
            extra = self._config.extra or {}

            api_key = extra.get("api_key") or os.environ.get("SWANLAB_API_KEY")
            log_dir = extra.get("log_dir") or os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            mode = extra.get("mode") or os.environ.get("SWANLAB_MODE", "cloud")

            # Login if API key is provided
            if api_key:
                swanlab.login(api_key)

            # Prepare config
            init_config = {"FRAMEWORK": "siirl_agentic"}
            if self._config.config:
                init_config.update(self._config.config)

            # Initialize SwanLab
            swanlab.init(
                project=self._config.project_name,
                experiment_name=self._config.experiment_name,
                config=init_config,
                logdir=log_dir,
                mode=mode,
            )

            self._initialized = True
            logger.success(f"SwanLab initialized: project={self._config.project_name}, mode={mode}")
            return True

        except ImportError:
            logger.warning("swanlab not installed. Install with: pip install swanlab")
            return False
        except Exception as e:
            logger.error(f"SwanLab initialization failed: {e}")
            return False

    def log(self, data: dict[str, NumericScalar], step: int) -> None:
        """
        Log scalar metrics to SwanLab.

        Args:
            data: Dictionary of metric names to values
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            self._swanlab.log(data, step=step)
        except Exception as e:
            logger.error(f"SwanLab log error: {e}")

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log text content to SwanLab.

        Args:
            tag: Tag for the text
            text: Text content
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            swanlab_text = self._swanlab.Text(text, caption=tag)
            self._swanlab.log({tag: swanlab_text}, step=step)
        except Exception as e:
            logger.error(f"SwanLab log_text error: {e}")

    def log_table(self, tag: str, columns: list[str], data: list[list[Any]], step: int) -> None:
        """
        Log tabular data to SwanLab as text entries.

        SwanLab doesn't have native table support, so we format as text.

        Args:
            tag: Tag for the table
            columns: Column names
            data: Rows of data
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            text_list = []
            for i, row in enumerate(data[:10]):  # Limit to 10 rows
                row_text = "\n---\n".join([f"{col}: {val}" for col, val in zip(columns, row, strict=False)])
                text_list.append(self._swanlab.Text(row_text, caption=f"row_{i+1}"))

            self._swanlab.log({tag: text_list}, step=step)
        except Exception as e:
            logger.error(f"SwanLab log_table error: {e}")

    def finish(self) -> None:
        """
        Finish the SwanLab experiment.
        """
        if self._initialized and self._swanlab:
            try:
                self._swanlab.finish()
                logger.info("SwanLab finished")
            except Exception as e:
                logger.error(f"SwanLab finish error: {e}")

        self._initialized = False
        self._swanlab = None
