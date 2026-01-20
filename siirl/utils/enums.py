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

from enum import Enum


class DistributedEnv(Enum):
    """Enumeration for distributed environment variable keys."""

    MASTER_ADDR = "MASTER_ADDR"
    MASTER_PORT = "MASTER_PORT"
    WORLD_SIZE = "WORLD_SIZE"
    RANK = "RANK"
    LOCAL_RANK = "LOCAL_RANK"
    WG_PREFIX = "WG_PREFIX"
    WG_BACKEND = "WG_BACKEND"
    RAY_LOCAL_WORLD_SIZE = "RAY_LOCAL_WORLD_SIZE"
    RAY_LOCAL_RANK = "RAY_LOCAL_RANK"
    DGA_PROCESS_GROUP = "DGA_PROCESS_GROUP"
