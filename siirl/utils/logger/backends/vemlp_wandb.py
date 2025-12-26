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
Volcengine ML Platform WandB Logger Backend

Provides integration with Volcengine ML Platform's WandB service.
"""

import os
from typing import Dict, List, Any
from loguru import logger
from .base import BackendConfig


class VemlpWandBBackend:
    """
    Volcengine ML Platform WandB logger backend.
    
    Features:
        - Lazy initialization
        - TensorBoard sync support
        - Volcengine authentication
    
    Configuration via BackendConfig.extra:
        - sync_tensorboard: Whether to sync with TensorBoard (default: True)
    
    Environment variables (required):
        - VOLC_ACCESS_KEY_ID: Volcengine access key ID
        - VOLC_SECRET_ACCESS_KEY: Volcengine secret access key
        - MLP_TRACKING_REGION: Volcengine ML Platform region
    """
    
    def __init__(self, config: BackendConfig):
        """
        Initialize Volcengine ML Platform WandB backend.
        
        Args:
            config: Backend configuration
        """
        self._name = "vemlp_wandb"
        self._config = config
        self._initialized = False
        self._wandb = None
    
    @property
    def name(self) -> str:
        return self._name
    
    def _ensure_initialized(self) -> bool:
        """
        Ensure Volcengine ML Platform WandB is initialized.
        
        Returns:
            True if successfully initialized, False otherwise
        """
        if self._initialized:
            return True
        
        try:
            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb
            
            # Check required environment variables
            ak = os.environ.get("VOLC_ACCESS_KEY_ID")
            sk = os.environ.get("VOLC_SECRET_ACCESS_KEY")
            region = os.environ.get("MLP_TRACKING_REGION")
            
            if not all([ak, sk, region]):
                missing = []
                if not ak:
                    missing.append("VOLC_ACCESS_KEY_ID")
                if not sk:
                    missing.append("VOLC_SECRET_ACCESS_KEY")
                if not region:
                    missing.append("MLP_TRACKING_REGION")
                logger.error(f"Missing required environment variables: {missing}")
                return False
            
            # Initialize Volcengine ML Platform
            volcengine_ml_platform.init(
                ak=ak,
                sk=sk,
                region=region,
            )
            
            # Get sync_tensorboard setting
            extra = self._config.extra or {}
            sync_tensorboard = extra.get("sync_tensorboard", True)
            
            # Initialize WandB
            vemlp_wandb.init(
                project=self._config.project_name,
                name=self._config.experiment_name,
                config=self._config.config,
                sync_tensorboard=sync_tensorboard,
            )
            
            self._wandb = vemlp_wandb
            self._initialized = True
            logger.success(f"Volcengine ML Platform WandB initialized: project={self._config.project_name}")
            return True
            
        except ImportError:
            logger.warning("volcengine_ml_platform not installed. Install with: pip install volcengine-ml-platform")
            return False
        except Exception as e:
            logger.error(f"Volcengine ML Platform WandB initialization failed: {e}")
            return False
    
    def log(self, data: Dict[str, float], step: int) -> None:
        """
        Log scalar metrics.
        
        Args:
            data: Dictionary of metric names to values
            step: Current training step
        """
        if not self._ensure_initialized():
            return
        
        try:
            self._wandb.log(data, step=step)
        except Exception as e:
            logger.error(f"Volcengine WandB log error: {e}")
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log text content.
        
        Args:
            tag: Tag for the text
            text: Text content
            step: Current training step
        """
        if not self._ensure_initialized():
            return
        
        try:
            # Use HTML for formatted text
            escaped_text = text.replace("<", "&lt;").replace(">", "&gt;")
            self._wandb.log({tag: self._wandb.Html(f"<pre>{escaped_text}</pre>")}, step=step)
        except Exception as e:
            logger.error(f"Volcengine WandB log_text error: {e}")
    
    def log_table(self, tag: str, columns: List[str], data: List[List[Any]], step: int) -> None:
        """
        Log tabular data.
        
        Args:
            tag: Tag for the table
            columns: Column names
            data: Rows of data
            step: Current training step
        """
        if not self._ensure_initialized():
            return
        
        try:
            table = self._wandb.Table(columns=columns, data=data)
            self._wandb.log({tag: table}, step=step)
        except Exception as e:
            logger.error(f"Volcengine WandB log_table error: {e}")
    
    def finish(self) -> None:
        """
        Finish the WandB run.
        """
        if self._initialized and self._wandb:
            try:
                self._wandb.finish(exit_code=0)
                logger.info("Volcengine ML Platform WandB finished")
            except Exception as e:
                logger.error(f"Volcengine WandB finish error: {e}")
        
        self._initialized = False
        self._wandb = None

