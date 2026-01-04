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
Logger Backend Registry

Provides a factory pattern for creating logger backends.

Usage:
    from siirl.utils.logger.backends import BackendRegistry, BackendConfig
    
    # Create a backend
    config = BackendConfig(project_name="project", experiment_name="exp")
    backend = BackendRegistry.create("wandb", config)
    
    # Register a custom backend
    BackendRegistry.register("custom", MyCustomBackend)
"""

from typing import Dict, Type
from .base import LoggerBackend, BackendConfig


class BackendRegistry:
    """
    Registry for logger backends.
    
    Supports dynamic registration and creation of backends.
    """
    
    _backends: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: Type) -> None:
        """
        Register a new backend.
        
        Args:
            name: Backend identifier
            backend_class: Backend class to register
        """
        cls._backends[name] = backend_class
    
    @classmethod
    def create(cls, name: str, config: BackendConfig) -> LoggerBackend:
        """
        Create a backend instance.
        
        Args:
            name: Backend identifier
            config: Backend configuration
            
        Returns:
            Backend instance
            
        Raises:
            ValueError: If backend is not registered
        """
        if name not in cls._backends:
            raise ValueError(f"Unknown backend: {name}. Available: {list(cls._backends.keys())}")
        return cls._backends[name](config)
    
    @classmethod
    def available(cls) -> list:
        """Get list of available backend names."""
        return list(cls._backends.keys())


def _register_builtin_backends():
    """Register built-in backends."""
    from .console import ConsoleBackend
    from .wandb import WandBBackend
    from .tensorboard import TensorBoardBackend
    from .mlflow import MLflowBackend
    from .swanlab import SwanLabBackend
    from .clearml import ClearMLBackend
    from .vemlp_wandb import VemlpWandBBackend
    
    BackendRegistry.register("console", ConsoleBackend)
    BackendRegistry.register("wandb", WandBBackend)
    BackendRegistry.register("tensorboard", TensorBoardBackend)
    BackendRegistry.register("mlflow", MLflowBackend)
    BackendRegistry.register("swanlab", SwanLabBackend)
    BackendRegistry.register("clearml", ClearMLBackend)
    BackendRegistry.register("vemlp_wandb", VemlpWandBBackend)


# Auto-register built-in backends on import
_register_builtin_backends()

__all__ = ["BackendRegistry", "BackendConfig", "LoggerBackend"]

