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

"""Pure helper functions for validate-reuse topology and partitioning."""

from siirl.utils.net_utils.net import SGLANG_DIST_INIT_START_PORT, SGLANG_HTTP_START_PORT
from siirl.worker.validate.reuse.constants import DIST_PORT_REUSE_OFFSET, HTTP_PORT_REUSE_OFFSET


def compute_validate_reuse_topology(train_gpus: int, tp_size: int, n_gpus_per_node: int) -> dict | None:
    if train_gpus <= 0 or tp_size <= 0 or n_gpus_per_node <= 0:
        return None

    gpus_per_rollout = min(tp_size, n_gpus_per_node)
    if tp_size % gpus_per_rollout != 0:
        return None

    num_tp_groups = train_gpus // tp_size
    if num_tp_groups == 0:
        return None

    usable_gpus = num_tp_groups * tp_size
    num_workers = usable_gpus // gpus_per_rollout
    rollout_per_tp_group = tp_size // gpus_per_rollout

    return {
        "usable_gpus": usable_gpus,
        "num_workers": num_workers,
        "num_tp_groups": num_tp_groups,
        "gpus_per_rollout": gpus_per_rollout,
        "rollout_per_tp_group": rollout_per_tp_group,
    }


def split_validate_samples(samples: list, num_workers: int) -> list[list]:
    if num_workers <= 0:
        return []
    shards = [[] for _ in range(num_workers)]
    for idx, sample in enumerate(samples):
        shards[idx % num_workers].append(sample)
    return shards


def split_validate_reuse_sync_workers(
    worker_infos: list[dict],
    trainer_node_ip: str,
    trainer_local_rank: int,
) -> tuple[list, list]:
    distributed_workers = []
    tensor_workers = []
    for info in worker_infos:
        worker = info["worker"]
        local_ranks = info.get("local_ranks") or []
        if info.get("node_ip") == trainer_node_ip and trainer_local_rank in local_ranks:
            tensor_workers.append(worker)
        else:
            distributed_workers.append(worker)
    return distributed_workers, tensor_workers


def rollout_to_train_step(rollout_step: int) -> int:
    return max(rollout_step - 1, 0)


def next_port_bases(port_window_idx: int, stride: int, cycle: int) -> tuple[int, int, int]:
    next_window_idx = (port_window_idx + 1) % max(1, cycle)
    offset = next_window_idx * max(1, stride)
    http_port_base = SGLANG_HTTP_START_PORT + HTTP_PORT_REUSE_OFFSET + offset
    dist_port_base = SGLANG_DIST_INIT_START_PORT + DIST_PORT_REUSE_OFFSET + offset
    return http_port_base, dist_port_base, next_window_idx
