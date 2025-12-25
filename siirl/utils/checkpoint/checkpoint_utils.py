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

import os
from typing import Optional


def find_latest_ckpt_path(checkpoint_dir: str) -> Optional[str]:
    """Find latest checkpoint based on tracker file."""
    tracker_file = os.path.join(checkpoint_dir, "latest_checkpointed_iteration.txt")

    if not os.path.exists(tracker_file):
        return None

    with open(tracker_file, "r") as f:
        global_step = f.read().strip()

    checkpoint_path = os.path.join(checkpoint_dir, f"global_step_{global_step}")
    return checkpoint_path if os.path.exists(checkpoint_path) else None
