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

"""Helpers to restore scalar metrics from weighted aggregation fields."""

from typing import Any

WEIGHTED_SUM_SUFFIX = "_weighted_sum"
WEIGHT_SUM_SUFFIX = "_weight_sum"


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            scalar = value.item()
        except Exception:
            return None
        if isinstance(scalar, int | float):
            return float(scalar)
    return None


def restore_weighted_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Convert weighted sum pairs into scalar metrics and remove helper fields.

    Example:
        actor/kl_loss_weighted_sum + actor/kl_loss_weight_sum -> actor/kl_loss
    """
    restored = dict(metrics)

    for key in list(metrics.keys()):
        if not key.endswith(WEIGHTED_SUM_SUFFIX):
            continue

        base_key = key[: -len(WEIGHTED_SUM_SUFFIX)]
        den_key = f"{base_key}{WEIGHT_SUM_SUFFIX}"
        if den_key not in metrics:
            continue

        num = _to_float(metrics.get(key))
        den = _to_float(metrics.get(den_key))
        if num is not None and den is not None and den > 0:
            restored[base_key] = num / den

        restored.pop(key, None)
        restored.pop(den_key, None)

    return restored
