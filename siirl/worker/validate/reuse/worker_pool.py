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

"""Worker pool lifecycle manager for validate-reuse rollout workers."""

import contextlib
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from siirl.worker.validate.reuse.topology import compute_validate_reuse_topology, next_port_bases


class ValidateReuseWorkerPool:
    """Manage init/destroy of validate-reuse rollout workers."""

    def __init__(self):
        self._workers = []
        self._tp0_workers = []
        self._tp0_worker_infos = []
        self._topology = None
        self._last_destroy_ts = 0.0
        self._port_window_idx = -1

    @property
    def workers(self) -> list:
        return self._workers

    @property
    def tp0_workers(self) -> list:
        return self._tp0_workers

    @property
    def tp0_worker_infos(self) -> list[dict]:
        return self._tp0_worker_infos

    def clear(self):
        self._workers = []
        self._tp0_workers = []
        self._tp0_worker_infos = []
        self._topology = None

    def destroy(self, *, graceful_shutdown_timeout_s: int):
        if not self._workers:
            self.clear()
            return

        import ray
        from loguru import logger

        shutdown_futures = [w.shutdown_engine.remote() for w in self._workers]
        try:
            ray.get(shutdown_futures, timeout=max(1, int(graceful_shutdown_timeout_s)))
        except Exception as e:
            logger.warning(f"[RolloutManager] Validate reuse engine shutdown failed: {e}")

        for i, worker in enumerate(self._workers):
            with contextlib.suppress(Exception):
                ray.kill(worker)
                logger.debug(f"[RolloutManager] Killed validate reuse worker {i}")

        self.clear()
        self._last_destroy_ts = time.time()

    def init_pool(
        self,
        *,
        enabled: bool,
        train_gpu_resources,
        tp_size: int,
        n_gpus_per_node: int,
        name_prefix: str,
        config,
        metric_worker,
        device_name: str,
        data_coordinator,
        create_worker_fn: Callable[..., Any],
        recreate_cooldown_s: float,
        port_stride: int,
        port_cycle: int,
        port_retry_slots: int,
    ) -> list:
        if not enabled or train_gpu_resources is None:
            self.clear()
            return []

        import ray
        from loguru import logger

        from siirl.worker.ray_utils import RayClassWithInitArgs
        from siirl.worker.rollout.rollout_worker import RolloutWorker

        cooldown_s = max(0.0, float(recreate_cooldown_s))
        if self._last_destroy_ts > 0 and cooldown_s > 0:
            elapsed = time.time() - self._last_destroy_ts
            if elapsed < cooldown_s:
                sleep_s = cooldown_s - elapsed
                logger.info(f"[RolloutManager] Waiting {sleep_s:.2f}s before recreating validate reuse workers")
                time.sleep(sleep_s)

        http_port_base, dist_port_base, next_window_idx = next_port_bases(
            port_window_idx=self._port_window_idx,
            stride=max(1, int(port_stride)),
            cycle=max(1, int(port_cycle)),
        )
        self._port_window_idx = next_window_idx
        retry_slots = max(1, int(port_retry_slots))
        logger.info(
            "[RolloutManager] Validate reuse port window "
            f"window_idx={self._port_window_idx} http_port_base={http_port_base} "
            f"dist_port_base={dist_port_base} retry_slots={retry_slots}"
        )

        train_pairs = list(zip(train_gpu_resources.indices, train_gpu_resources.local_ranks, strict=False))
        topology = compute_validate_reuse_topology(
            len(train_pairs),
            tp_size,
            n_gpus_per_node,
        )
        if topology is None:
            logger.info("[RolloutManager] Validate GPU reuse disabled: no full TP group available on training GPUs")
            self.clear()
            return []

        self._topology = topology
        use_gpus = topology["usable_gpus"]
        num_workers = topology["num_workers"]
        num_tp_groups = topology["num_tp_groups"]
        gpus_per_rollout = topology["gpus_per_rollout"]
        rollout_per_tp_group = topology["rollout_per_tp_group"]

        selected_pairs = train_pairs[:use_gpus]
        indices = [idx for idx, _ in selected_pairs]
        local_ranks = [lr for _, lr in selected_pairs]

        reuse_prefix = f"{name_prefix}_valreuse"
        reuse_ray_class = RayClassWithInitArgs(ray.remote(RolloutWorker), config, num_tp_groups, metric_worker)

        workers = []
        for worker_idx in range(num_workers):
            first_gpu_idx = worker_idx * gpus_per_rollout
            bundle_idx = indices[first_gpu_idx]
            local_rank = local_ranks[first_gpu_idx]
            worker = create_worker_fn(
                rank=worker_idx,
                local_rank=local_rank,
                bundle_idx=bundle_idx,
                num_gpus=0,
                device_name=device_name,
                num_cpus=0,
                world_size=num_workers,
                worker_prefix=reuse_prefix,
                rollout_ray_class=reuse_ray_class,
            )
            workers.append(worker)

        dist_init_addrs = {}
        configs = []
        for worker_idx in range(num_workers):
            tp_group_idx = worker_idx // rollout_per_tp_group
            node_rank = worker_idx % rollout_per_tp_group
            first_gpu_idx = worker_idx * gpus_per_rollout
            base_gpu_id = local_ranks[first_gpu_idx]
            worker_local_ranks = local_ranks[first_gpu_idx : first_gpu_idx + gpus_per_rollout]
            if rollout_per_tp_group > 1:
                if node_rank == 0:
                    dist_init_start_port = dist_port_base + tp_group_idx
                    dist_init_addr = ray.get(workers[worker_idx].get_ip_port.remote(start_port=dist_init_start_port))
                    dist_init_addrs[tp_group_idx] = dist_init_addr
                else:
                    dist_init_addr = dist_init_addrs[tp_group_idx]
            else:
                dist_init_addr = None
            configs.append(
                {
                    "worker_idx": worker_idx,
                    "base_gpu_id": base_gpu_id,
                    "node_rank": node_rank,
                    "nnodes": rollout_per_tp_group,
                    "dist_init_addr": dist_init_addr,
                    "is_tp0": node_rank == 0,
                    "worker_local_ranks": worker_local_ranks,
                }
            )

        worker_ips = {}
        node_workers = defaultdict(list)
        for cfg in configs:
            worker = workers[cfg["worker_idx"]]
            ip = ray.get(worker.get_ip.remote())
            worker_ips[cfg["worker_idx"]] = ip
            node_workers[ip].append((worker, cfg))

        worker_reserved_ports = {}
        for _, workers_on_node in node_workers.items():
            first_worker = workers_on_node[0][0]
            worker_count = len(workers_on_node)
            ports = ray.get(first_worker.allocate_ports.remote(start_port=http_port_base, count=worker_count * retry_slots))
            for i, (_, cfg) in enumerate(workers_on_node):
                worker_reserved_ports[cfg["worker_idx"]] = [ports[i + j * worker_count] for j in range(retry_slots)]

        init_futures = []
        for cfg in configs:
            worker = workers[cfg["worker_idx"]]
            reserved_ports = worker_reserved_ports[cfg["worker_idx"]]
            init_futures.append(
                worker.init_engine.remote(
                    rank=cfg["worker_idx"],
                    dist_init_addr=cfg["dist_init_addr"],
                    ip=worker_ips[cfg["worker_idx"]],
                    port=reserved_ports[0],
                    base_gpu_id=cfg["base_gpu_id"],
                    node_rank=cfg["node_rank"],
                    nnodes=cfg["nnodes"],
                )
            )
        ray.get(init_futures)

        launch_futures = []
        for cfg in configs:
            worker = workers[cfg["worker_idx"]]
            reserved_ports = worker_reserved_ports[cfg["worker_idx"]]
            launch_futures.append(worker.launch_server.remote(max_retries=len(reserved_ports), reserved_ports=reserved_ports))
        ray.get(launch_futures)

        tp0_workers = []
        tp0_infos = []
        for cfg in configs:
            if cfg["is_tp0"]:
                worker = workers[cfg["worker_idx"]]
                tp0_workers.append(worker)
                tp0_infos.append(
                    {
                        "worker": worker,
                        "node_ip": worker_ips[cfg["worker_idx"]],
                        "local_ranks": cfg["worker_local_ranks"],
                    }
                )
        ray.get([w.init_validate_executor.remote(data_coordinator, num_tp_groups) for w in tp0_workers])

        logger.info(f"[RolloutManager] Validate GPU reuse enabled: {use_gpus} train GPUs, " f"{len(tp0_workers)} extra TP0 workers")
        self._workers = workers
        self._tp0_workers = tp0_workers
        self._tp0_worker_infos = tp0_infos
        return tp0_workers
