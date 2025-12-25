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
Logger Backend Protocol and Configuration

Defines the interface that all logger backends must implement.
"""

from typing import Protocol, Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BackendConfig:
    """
    Configuration for logger backends.
    
    Attributes:
        project_name: Project name (e.g., "siirl_agentic")
        experiment_name: Experiment name (e.g., "gsm8k_grpo_exp1")
        config: Training configuration dictionary to log
        extra: Backend-specific configuration (e.g., {"proxy": "..."} for wandb)
    """
    project_name: str
    experiment_name: str
    config: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)


class LoggerBackend(Protocol):
    """
    Protocol defining the interface for logger backends.
    
    All backends must implement these methods to ensure consistency.
    
    Methods:
        name: Backend identifier property
        log: Log scalar metrics
        log_text: Log text content
        log_table: Log tabular data
        finish: Clean up resources
    """
    
    @property
    def name(self) -> str:
        """Backend name identifier."""
        ...
    
    def log(self, data: Dict[str, float], step: int) -> None:
        """
        Log scalar metrics.
        
        Args:
            data: Dictionary of metric names to values
            step: Current training step
        """
        ...
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log text content.
        
        Args:
            tag: Tag/label for the text
            text: Text content to log
            step: Current training step
        """
        ...
    
    def log_table(self, tag: str, columns: List[str], data: List[List[Any]], step: int) -> None:
        """
        Log tabular data.
        
        Args:
            tag: Tag/label for the table
            columns: List of column names
            data: List of rows, each row is a list of values
            step: Current training step
        """
        ...
    
    def finish(self) -> None:
        """
        Clean up resources and close connections.
        
        Should be called when logging is complete.
        """
        ...

