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

import math
from collections.abc import Mapping
from typing import Literal


def resolve_train_server_concurrency(config) -> int:
    """Resolve the primary local concurrency knob."""
    configured = int(getattr(config.rollout, "train_server_concurrency", 0))
    if configured > 0:
        return configured
    # Safe default for single-node and multi-node bring-up.
    return 64


def resolve_max_num_seqs(config) -> int:
    """Resolve SGLang max running requests with optional auto mode."""
    configured = int(getattr(config.rollout, "max_num_seqs", 0))
    if configured > 0:
        return configured
    # Auto: preserve a small queue headroom above local train concurrency.
    return max(1, resolve_train_server_concurrency(config) * 4)


def resolve_rollout_concurrency(
    config,
    *,
    phase: Literal["train", "validate"],
    use_router: bool,
) -> Mapping[str, int | str | bool]:
    """Resolve effective client-side request concurrency with a hard scheduler cap."""
    max_num_seqs = resolve_max_num_seqs(config)
    train_concurrency = resolve_train_server_concurrency(config)
    rollout_gpus = max(1, int(getattr(config.trainer, "rollout_gpus", 1)))
    tp_size = max(1, int(getattr(config.rollout, "tensor_model_parallel_size", 1)))
    num_engines = max(1, rollout_gpus // tp_size)

    if use_router:
        # Auto: saturate max_num_seqs across all engines.
        base_key = "auto_router_concurrency"
        base = max(1, math.ceil(max_num_seqs / num_engines))
        resolved = base * num_engines
    else:
        if phase not in {"train", "validate"}:
            raise ValueError(f"Unsupported phase: {phase}")
        # Keep local validate concurrency aligned with train by default.
        base_key = "train_server_concurrency"
        base = train_concurrency
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
