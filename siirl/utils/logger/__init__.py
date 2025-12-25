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
siiRL-agentic Logger Module

Provides unified experiment tracking and logging functionality with support
for multiple backends (console, wandb, tensorboard).

Usage:
    # Basic logging with MetricTracker
    from siirl.utils.logger import MetricTracker, GenerationSample
    
    tracker = MetricTracker(
        project_name="siirl_agentic",
        experiment_name="gsm8k_grpo",
        backends=["console", "wandb"],
        config={"lr": 1e-4, "batch_size": 32}
    )
    
    # Log metrics
    tracker.log({"loss": 0.5, "accuracy": 0.9}, step=100)
    
    # Log generation samples
    samples = [GenerationSample("2+2=?", "4", 1.0)]
    tracker.log_generation(samples, step=100)
    
    # Close when done
    tracker.finish()
    
    # Or use context manager
    with MetricTracker("project", "exp", ["console"]) as tracker:
        tracker.log({"loss": 0.5}, step=1)
    
    # Custom backend registration
    from siirl.utils.logger import BackendRegistry
    BackendRegistry.register("custom", MyCustomBackend)

For distributed metrics collection, see siirl.utils.metrics module.
"""

from .logging_utils import set_basic_config
from .tracker import MetricTracker, GenerationSample
from .backends import BackendRegistry, BackendConfig

__all__ = [
    # Logging configuration
    "set_basic_config",
    # Main tracker class
    "MetricTracker",
    "GenerationSample",
    # Backend extensibility
    "BackendRegistry",
    "BackendConfig",
]

