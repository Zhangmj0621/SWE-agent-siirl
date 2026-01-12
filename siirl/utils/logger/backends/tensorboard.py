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
TensorBoard Logger Backend

Provides integration with TensorBoard for experiment visualization.
"""

import os
from typing import Any

from loguru import logger

from .base import BackendConfig


class TensorBoardBackend:
    """
    TensorBoard logger backend.

    Features:
        - Lazy initialization
        - Automatic directory creation
        - Scalar, text, and table (as markdown) logging

    Configuration via BackendConfig.extra:
        - log_dir: Base directory for TensorBoard logs (default: "tensorboard_logs")
    """

    def __init__(self, config: BackendConfig):
        """
        Initialize TensorBoard backend.

        Args:
            config: Backend configuration with optional log_dir setting
        """
        self._name = "tensorboard"
        self._config = config
        self._writer = None

        # Get log directory from config or use default
        if config.extra and "log_dir" in config.extra:
            self._log_dir = config.extra["log_dir"]
        else:
            self._log_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_logs")

    @property
    def name(self) -> str:
        return self._name

    def _ensure_initialized(self) -> bool:
        """
        Ensure TensorBoard writer is initialized.

        Returns:
            True if successfully initialized, False otherwise
        """
        if self._writer is not None:
            return True

        try:
            from torch.utils.tensorboard import SummaryWriter

            # Create log path with experiment name
            log_path = os.path.join(self._log_dir, self._config.experiment_name)
            os.makedirs(log_path, exist_ok=True)

            self._writer = SummaryWriter(log_dir=log_path)
            logger.info(f"TensorBoard logs: {log_path}")
            return True

        except ImportError:
            logger.warning("tensorboard not installed. Install with: pip install tensorboard")
            return False
        except Exception as e:
            logger.error(f"TensorBoard initialization failed: {e}")
            return False

    def log(self, data: dict[str, float], step: int) -> None:
        """
        Log scalar metrics to TensorBoard.

        Args:
            data: Dictionary of metric names to values
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            for key, value in data.items():
                self._writer.add_scalar(key, value, step)
        except Exception as e:
            logger.error(f"TensorBoard log error: {e}")

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log text content to TensorBoard.

        Args:
            tag: Tag for the text
            text: Text content
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            self._writer.add_text(tag, text, step)
        except Exception as e:
            logger.error(f"TensorBoard log_text error: {e}")

    def log_table(self, tag: str, columns: list[str], data: list[list[Any]], step: int) -> None:
        """
        Log tabular data to TensorBoard as markdown.

        TensorBoard doesn't natively support tables, so we convert to markdown format.

        Args:
            tag: Tag for the table
            columns: Column names
            data: Rows of data
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            # Build markdown table
            md_lines = []

            # Header
            md_lines.append("| " + " | ".join(columns) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

            # Rows (limit for display)
            for row in data[:20]:
                # Truncate cell content and escape pipes
                cells = [str(v)[:50].replace("|", "\\|") for v in row]
                md_lines.append("| " + " | ".join(cells) + " |")

            if len(data) > 20:
                md_lines.append(f"| ... | {len(data) - 20} more rows | ... |")

            markdown_text = "\n".join(md_lines)
            self._writer.add_text(tag, markdown_text, step)

        except Exception as e:
            logger.error(f"TensorBoard log_table error: {e}")

    def finish(self) -> None:
        """
        Flush and close the TensorBoard writer.
        """
        if self._writer:
            try:
                self._writer.flush()
                self._writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.error(f"TensorBoard finish error: {e}")

        self._writer = None
