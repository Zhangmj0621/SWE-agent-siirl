# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
# Copyright 2025, Infrawaves. All rights reserved.
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
Dynamic batching module.

Intelligently partition micro-batches based on sequence lengths to optimize GPU utilization.

Features:
- FLOPs-based workload estimation
- Karmarkar-Karp near-optimal partitioning
- Pipeline bubble optimization
- Cross-DP rank synchronization
"""

from __future__ import annotations

import heapq
from itertools import chain

import torch
from tensordict import TensorDict
from torch import distributed as dist


def calculate_workload(seq_lens: torch.Tensor) -> torch.Tensor:
    """Estimate transformer attention computational workload.

    Based on: FLOPs = 12*h^2*s + 2*h*s^2 (for 7B model h=4096)
    Simplified: workload = 24576*s + s^2
    """
    return 24576 * seq_lens + seq_lens**2


def _karmarkar_karp(values: list[int], k: int, equal_size: bool) -> list[list[int]]:
    """Karmarkar-Karp multi-way partitioning (Largest Differencing Method).

    Partition n numbers into k groups with balanced sums.
    """
    n = len(values)
    assert n >= k, f"num_items({n}) < num_partitions({k})"
    if equal_size:
        assert n % k == 0, f"equal_size requires n({n}) divisible by k({k})"

    # Sort by value, keep original indices
    sorted_items = sorted(enumerate(values), key=lambda x: x[1])

    # State: (negative spread, group sums tuple, group index lists)
    def make_state(groups: list[list[tuple[int, int]]]):
        sums = tuple(sum(v for _, v in g) for g in groups)
        spread = max(sums) - min(sums) if sums else 0
        return (-spread, sums, groups)

    # Initialize priority queue
    pq = []
    if equal_size:
        for offset in range(0, n, k):
            groups = [[(sorted_items[offset + i][0], sorted_items[offset + i][1])] for i in range(k)]
            heapq.heappush(pq, make_state(groups))
    else:
        for idx, val in sorted_items:
            groups = [[(idx, val)]] + [[] for _ in range(k - 1)]
            heapq.heappush(pq, make_state(groups))

    # Iteratively merge states
    while len(pq) > 1:
        _, _, groups1 = heapq.heappop(pq)
        _, _, groups2 = heapq.heappop(pq)

        # Sort by sum descending, then merge in reverse order (large with small)
        sums1 = [(sum(v for _, v in g), i) for i, g in enumerate(groups1)]
        sums2 = [(sum(v for _, v in g), i) for i, g in enumerate(groups2)]
        sums1.sort(reverse=True)
        sums2.sort(reverse=True)

        merged = [[] for _ in range(k)]
        for rank, ((_, i1), (_, i2)) in enumerate(zip(sums1, reversed(sums2), strict=False)):
            merged[rank] = groups1[i1] + groups2[i2]

        heapq.heappush(pq, make_state(merged))

    # Extract final partition indices
    _, _, final_groups = pq[0]
    return [sorted(idx for idx, _ in g) for g in final_groups]


def balanced_partition(values: list[int], k: int, equal_size: bool = False) -> list[list[int]]:
    """Partition values into k balanced groups."""
    partitions = _karmarkar_karp(values, k, equal_size)

    # Validate completeness
    all_indices = set(chain.from_iterable(partitions))
    assert all_indices == set(range(len(values))), "Incomplete partition"
    assert all(len(p) > 0 for p in partitions), "Empty partition exists"

    return partitions


def rearrange_micro_batches(
    batch: TensorDict,
    max_token_len: int,
    dp_group=None,
    vpp_size: int | None = None,
    sync_micro_num: bool = True,
    optimize_bubble: bool = True,
) -> tuple[list[TensorDict], list[list[int]]]:
    """Dynamically partition batch into micro-batches.

    Args:
        batch: Input batch with attention_mask
        max_token_len: Max total tokens per micro-batch
        dp_group: Data parallel communication group
        vpp_size: Virtual pipeline parallel size
        sync_micro_num: Sync micro-batch count across DP ranks
        optimize_bubble: Optimize ordering to reduce pipeline bubbles

    Returns:
        (micro_batches, partitions): Partitioned micro-batches and original indices
    """
    attention_mask = batch["attention_mask"]
    seq_lens = attention_mask.sum(dim=1).long()
    batch_size = len(seq_lens)
    # Use per-sample effective length instead of padded width so CP/padding layouts
    # don't trigger false positives for the budget feasibility check.
    max_seq_len = int(seq_lens.max().item()) if batch_size > 0 else 0

    assert max_token_len >= max_seq_len, f"max_token_len({max_token_len}) < max_seq_len({max_seq_len})"

    # Determine number of micro-batches
    total_tokens = seq_lens.sum().item()
    num_micro = min(batch_size, -(-total_tokens // max_token_len))

    # Sync across DP ranks
    if sync_micro_num and dist.is_initialized() and dp_group is not None:
        num_micro_t = torch.tensor([num_micro], device=seq_lens.device)
        dist.all_reduce(num_micro_t, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro = num_micro_t.item()

    # Align to vpp_size
    if vpp_size is not None:
        num_micro = -(-num_micro // vpp_size) * vpp_size

    num_micro = min(num_micro, batch_size)

    # Workload-based balanced partition
    workloads = calculate_workload(seq_lens).cpu().tolist()
    partitions = balanced_partition(workloads, num_micro, equal_size=False)

    # Pipeline bubble optimization: large workloads in middle, small at ends
    if optimize_bubble and len(partitions) > 1:
        partitions.sort(key=lambda p: sum(workloads[i] for i in p), reverse=True)
        partitions = partitions[::2][::-1] + partitions[1::2]

    # Build micro-batches
    micro_batches = [batch[torch.tensor(p, dtype=torch.long)] for p in partitions]

    return micro_batches, partitions


def restore_batch_order(data: torch.Tensor, partitions: list[list[int]]) -> torch.Tensor:
    """Restore original batch order after dynamic batching."""
    flat_indices = list(chain.from_iterable(partitions))
    reverse_map = [0] * len(flat_indices)
    for new_idx, old_idx in enumerate(flat_indices):
        reverse_map[old_idx] = new_idx

    return data[torch.tensor(reverse_map, dtype=torch.long, device=data.device)]
