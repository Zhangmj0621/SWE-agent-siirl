# Copyright 2026, Shanghai Innovation Institute. All rights reserved.
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

from collections.abc import Mapping
from typing import Literal


def resolve_rollout_concurrency(
    config,
    *,
    phase: Literal["train", "validate"],
    use_router: bool,
) -> Mapping[str, int | str | bool]:
    """Resolve effective client-side request concurrency with a hard scheduler cap."""
    max_num_seqs = max(1, int(getattr(config.rollout, "max_num_seqs", 1)))
    rollout_gpus = max(1, int(getattr(config.trainer, "rollout_gpus", 1)))
    tp_size = max(1, int(getattr(config.rollout, "tensor_model_parallel_size", 1)))
    num_engines = max(1, rollout_gpus // tp_size)

    if use_router:
        base_key = "server_concurrency"
        base = max(1, int(getattr(config.rollout, base_key, 1)))
        resolved = base * num_engines
    else:
        if phase == "train":
            base_key = "train_server_concurrency"
            base = max(1, int(getattr(config.rollout, base_key, 256)))
        elif phase == "validate":
            base_key = "validate_server_concurrency"
            base = max(1, int(getattr(config.rollout, base_key, 256)))
        else:
            raise ValueError(f"Unsupported phase: {phase}")
        resolved = base

    effective = min(resolved, max_num_seqs)
    return {
        "phase": phase,
        "use_router": bool(use_router),
        "base_key": base_key,
        "base": base,
        "num_engines": num_engines,
        "resolved": resolved,
        "max_num_seqs": max_num_seqs,
        "effective": effective,
    }
