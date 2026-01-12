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
MLflow Logger Backend

Provides integration with MLflow for experiment tracking.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from .base import BackendConfig


class MLflowBackend:
    """
    MLflow logger backend.

    Features:
        - Lazy initialization
        - Automatic experiment creation
        - Artifact logging for tables/generations

    Configuration via BackendConfig.extra:
        - tracking_uri: MLflow tracking server URI (or use MLFLOW_TRACKING_URI env var)

    Environment variables:
        - MLFLOW_TRACKING_URI: MLflow tracking server URI
    """

    def __init__(self, config: BackendConfig):
        """
        Initialize MLflow backend.

        Args:
            config: Backend configuration
        """
        self._name = "mlflow"
        self._config = config
        self._initialized = False
        self._mlflow = None

    @property
    def name(self) -> str:
        return self._name

    def _ensure_initialized(self) -> bool:
        """
        Ensure MLflow is initialized.

        Returns:
            True if successfully initialized, False otherwise
        """
        if self._initialized:
            return True

        try:
            import mlflow

            self._mlflow = mlflow

            # Set tracking URI from config or environment
            tracking_uri = None
            if self._config.extra:
                tracking_uri = self._config.extra.get("tracking_uri")
            if not tracking_uri:
                tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            # Create/set experiment (project_name maps to experiment in MLflow)
            experiment = mlflow.set_experiment(self._config.project_name)

            # Start run with experiment_name as run_name
            mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=self._config.experiment_name,
            )

            # Log config as params
            if self._config.config:
                params = self._flatten_config(self._config.config)
                # MLflow has a limit on param value length
                params = {k: str(v)[:250] for k, v in params.items()}
                mlflow.log_params(params)

            self._initialized = True
            logger.success(f"MLflow initialized: experiment={self._config.project_name}, run={self._config.experiment_name}")
            return True

        except ImportError:
            logger.warning("mlflow not installed. Install with: pip install mlflow")
            return False
        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}")
            return False

    def _flatten_config(self, config: dict, prefix: str = "") -> dict[str, Any]:
        """Flatten nested config dict for MLflow params."""
        items = {}
        for k, v in config.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_config(v, key))
            else:
                items[key] = v
        return items

    def log(self, data: dict[str, float], step: int) -> None:
        """
        Log scalar metrics to MLflow.

        Args:
            data: Dictionary of metric names to values
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            # MLflow doesn't allow '@' in metric names
            sanitized = {k.replace("@", "_at_"): v for k, v in data.items()}
            self._mlflow.log_metrics(metrics=sanitized, step=step)
        except Exception as e:
            logger.error(f"MLflow log error: {e}")

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log text content to MLflow as artifact.

        Args:
            tag: Tag for the text
            text: Text content
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = Path(tmp_dir) / f"{tag.replace('/', '_')}_step{step}.txt"
                file_path.write_text(text)
                self._mlflow.log_artifact(str(file_path))
        except Exception as e:
            logger.error(f"MLflow log_text error: {e}")

    def log_table(self, tag: str, columns: list[str], data: list[list[Any]], step: int) -> None:
        """
        Log tabular data to MLflow as JSON artifact.

        Args:
            tag: Tag for the table
            columns: Column names
            data: Rows of data
            step: Current training step
        """
        if not self._ensure_initialized():
            return

        try:
            # Convert to list of dicts
            records = [dict(zip(columns, row, strict=False)) for row in data]

            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = Path(tmp_dir) / f"{tag.replace('/', '_')}_step{step}.json"
                with open(file_path, "w") as f:
                    json.dump(records, f, indent=2, default=str)
                self._mlflow.log_artifact(str(file_path))
        except Exception as e:
            logger.error(f"MLflow log_table error: {e}")

    def finish(self) -> None:
        """
        End the MLflow run.
        """
        if self._initialized and self._mlflow:
            try:
                self._mlflow.end_run()
                logger.info("MLflow run ended")
            except Exception as e:
                logger.error(f"MLflow finish error: {e}")

        self._initialized = False
        self._mlflow = None
