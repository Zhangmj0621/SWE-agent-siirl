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
Weights & Biases Logger Backend

Provides integration with wandb for experiment tracking.
"""

from typing import Any

from loguru import logger

from .base import BackendConfig


class WandBBackend:
    """
    Weights & Biases logger backend.

    Features:
        - Lazy initialization (connects only when first log is called)
        - Proxy support for corporate environments
        - Table logging for generation samples
        - Graceful error handling

    Configuration via BackendConfig.extra:
        - proxy: HTTPS proxy URL (e.g., "http://proxy.example.com:8080")
    """

    def __init__(self, config: BackendConfig):
        """
        Initialize WandB backend.

        Args:
            config: Backend configuration with optional proxy settings
        """
        self._name = "wandb"
        self._config = config
        self._run = None
        self._initialized = False
        self._table_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def _ensure_initialized(self) -> bool:
        """
        Ensure wandb is initialized, connecting lazily on first use.

        Returns:
            True if successfully initialized, False otherwise
        """
        if self._initialized:
            return True

        try:
            import wandb

            # Configure settings
            settings = None
            if self._config.extra:
                proxy = self._config.extra.get("proxy")
                if proxy:
                    settings = wandb.Settings(https_proxy=proxy)

            # Initialize wandb run
            self._run = wandb.init(
                project=self._config.project_name,
                name=self._config.experiment_name,
                config=self._config.config,
                settings=settings,
                reinit=True,
            )

            self._initialized = True
            logger.success(f"WandB initialized: {self._run.url}")
            return True

        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            return False
        except Exception as e:
            logger.error(f"WandB initialization failed: {e}")
            return False

    def log(self, data: dict[str, float], step: int) -> None:
        """
        Log scalar metrics to wandb.

        Args:
            data: Dictionary of metric names to values
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            import wandb

            wandb.log(data, step=step)
        except Exception as e:
            logger.error(f"WandB log error: {e}")

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log text content to wandb as HTML.

        Args:
            tag: Tag for the text
            text: Text content
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            import wandb

            # Escape HTML and wrap in pre tag for formatting
            escaped_text = text.replace("<", "&lt;").replace(">", "&gt;")
            wandb.log({tag: wandb.Html(f"<pre>{escaped_text}</pre>")}, step=step)
        except Exception as e:
            logger.error(f"WandB log_text error: {e}")

    def log_table(self, tag: str, columns: list[str], data: list[list[Any]], step: int) -> None:
        """
        Log tabular data to wandb.

        Creates a wandb.Table that accumulates data across steps.

        Args:
            tag: Tag for the table
            columns: Column names
            data: Rows of data
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            import wandb

            # WandB tables need special handling for appending
            # See: https://github.com/wandb/wandb/issues/2981
            full_columns = ["step"] + columns

            if tag not in self._table_cache:
                self._table_cache[tag] = wandb.Table(columns=full_columns)

            # Create new table with existing data (workaround for wandb limitation)
            old_data = self._table_cache[tag].data
            new_table = wandb.Table(columns=full_columns, data=old_data)

            # Add new rows
            for row in data:
                new_table.add_data(step, *row)

            # Log and update cache
            wandb.log({tag: new_table}, step=step)
            self._table_cache[tag] = new_table

        except Exception as e:
            logger.error(f"WandB log_table error: {e}")

    def finish(self) -> None:
        """
        Finish the wandb run and clean up.
        """
        if self._initialized and self._run:
            try:
                self._run.finish(exit_code=0)
                logger.info("WandB run finished")
            except Exception as e:
                logger.error(f"WandB finish error: {e}")

        self._initialized = False
        self._run = None
        self._table_cache.clear()
