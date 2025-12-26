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
SiiRL - Shanghai Innovation Institute Reinforcement Learning Framework

This package automatically configures logging when imported.
All Ray actors and processes that import from siirl will have
consistent logging configuration.

Log configuration can be customized via environment variables:
- LOGURU_LEVEL: Log level (default: INFO)
- SIIRL_LOG_DIRECTORY: Directory for log files (default: siirl_logs)
- SIIRL_LOGGING_FILENAME: Prefix for log filenames (default: siirl)
"""

from siirl.utils.logger.logging_utils import set_basic_config

# Automatically configure logging when siirl is imported
# This ensures consistent logging across all processes (main process and Ray actors)
set_basic_config()

__all__ = []
