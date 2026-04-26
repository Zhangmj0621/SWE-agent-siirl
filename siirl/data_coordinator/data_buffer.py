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

import asyncio
import copy
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import loguru
import ray

from siirl.data_coordinator.dataloader import DataLoaderNode
from siirl.data_coordinator.sample import Dict2Samples, SampleGroup, SampleInfo, preprocess_dataloader
from siirl.params.training_args import SiiRLArguments
from siirl.utils.model_utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions


@ray.remote(max_concurrency=1, concurrency_groups={"dataloader": 1}, num_cpus=2)
class DataCoordinator:
    """
    A globally unique central Actor responsible for coordinating data producers (RolloutWorkers)
    and consumers (Trainers). It does not store the actual sample data, only the sample
    metadata (SampleInfo) and object references (ObjectRef). This allows it to implement
    complex global sampling strategies at a very low cost.

    Important change!!!
    To ensure rollout worker not blocked by main loop, we expect to prefetch some sample from dataloader to pending_queue.
    Specifically, we prefetch global bsz samples to pending_queue.
    This strategy is designed for both colocation and disaggregation since it doesn't cause off-policy.
    """

    def __init__(self, nnodes: int, ppo_mini_batch_size: int, world_size: int, rollout_n: int):
        self.nnodes = nnodes
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.world_size = world_size
        self.rollout_n = max(1, int(rollout_n))
        # Use a deque to store tuples of metadata and references for efficient FIFO operations
        self._sample_queue: deque[tuple[SampleInfo, ray.ObjectRef]] = deque()
        self._put_counter = 0  # Used for round-robin buffer selection
        self._batch_wait_log_counter = 0  # Counter to throttle "waiting for samples" logs
        self.lock = asyncio.Lock()
        loguru.logger.info("Global DataCoordinator initialized.")
        self._cache = []

        # # dataloader
        self.pending_queue = asyncio.Queue()
        self.dataloader_queue = asyncio.Queue()
        self.dataloader_val_queue = asyncio.Queue()
        self.dataloader = None

        # Background task used to move samples from pending_queue -> dataloader_queue.
        self._prepare_data_task: asyncio.Task | None = None
        self._prepare_data_event = asyncio.Event()
        self._prepare_data_stop = False
        self._prepare_data_reserve = 0

        # Used to avoid uid = 0 in async mode
        self._next_train_uid = 0

        # Shared cancel queue for partial-rollout samples aborted by weight sync.
        # Any rollout worker may push to / pull from this queue — partial samples
        # do NOT need to return to the original worker since flush_cache happens
        # globally during weight sync, so KV/prefix cache cannot be reused anyway.
        # Hoisting this to the coordinator load-balances partial work across workers.
        self._cancel_queue: asyncio.Queue = asyncio.Queue()

        # Per-uid group container. Each group has rollout_n fixed slots; replicas
        # are written into their slot by replica_index on ``put``. When every slot
        # is filled the group is flushed to ``_sample_queue`` as a contiguous run
        # in replica-index order.
        self._pending_groups: dict[str, SampleGroup] = {}

    async def put(self, sample_info: SampleInfo, sample_ref: Any):
        """
        Called by a RolloutWorker with one finished rollout replica.

        Writes the replica into its pre-allocated slot in _pending_groups[uid]
        (see SampleGroup). When every slot of the group is filled, the group
        is atomically flushed to _sample_queue in replica-index order so the
        trainer sees replica 0, 1, ..., rollout_n-1 as a contiguous run.

        Args:
            sample_info: Metadata about the sample (must carry uid and replica_index)
            sample_ref: Ray ObjectRef or the actual sample data
        """
        # Due to Ray's small object optimization, an ObjectRef passed by the client
        # might be automatically resolved to its actual value. Here, we ensure that
        # we are always handling an ObjectRef.
        if not isinstance(sample_ref, ray.ObjectRef):
            sample_ref = ray.put(sample_ref)

        uid = sample_info.uid
        replica_index = sample_info.replica_index
        async with self.lock:
            group = self._pending_groups.get(uid)
            if group is None:
                # Partial-rollout resume may arrive after the process that created the
                # group died; rebuild a group lazily so we don't drop the replica.
                group = SampleGroup(uid=uid, rollout_n=self.rollout_n)
                self._pending_groups[uid] = group
            group.replicas[replica_index] = (sample_info, sample_ref)
            if group.is_complete():
                self._sample_queue.extend(self._pending_groups.pop(uid).replicas)

    async def get_batch(
        self,
        batch_size: int,
        dp_rank: int,
        filter_plugin: Callable[[SampleInfo], bool] | None = None,
        balance_partitions: int | None = None,
        min_version: int | None = None,
    ) -> list[ray.ObjectRef]:
        """Called by a Trainer to get a batch of sample ObjectRefs.

        Supports an optional filter plugin to implement custom sampling logic, and an
        optional length balancing feature.

        Args:
            batch_size: The requested batch size.
            filter_plugin: optional filters function for custom sampling logic.
            balance_partitions: If specified, the returned samples will be optimized
                              for even distribution among the given number of workers,
                              balancing the sum of sequence lengths for each worker.
                              Defaults to None (no length balancing).
            min_version: If specified, samples with weight_version < min_version will
                        be discarded. This supports off-policy training by removing
                        stale data generated by old model versions.

        Returns:
            A list of sample ObjectRefs. If length balancing is enabled, the order
            of samples will be optimized.
        """
        async with self.lock:
            # Clean up stale samples if min_version is specified
            if min_version is not None:
                stale_count = 0
                new_queue = deque()
                for item in self._sample_queue:
                    sample_info, sample_ref = item
                    if sample_info.weight_version >= min_version:
                        new_queue.append(item)
                    else:
                        stale_count += 1
                if stale_count > 0:
                    loguru.logger.debug(f"Discarded {stale_count} stale samples with version < {min_version}")
                self._sample_queue = new_queue
            # No filter plugin, use efficient FIFO
            global_batch_size = batch_size * balance_partitions
            if len(self._cache) > 0:
                res = self._cache[dp_rank]
                return res
            if not filter_plugin:
                if len(self._sample_queue) < global_batch_size:
                    self._batch_wait_log_counter += 1
                    if self._batch_wait_log_counter == 1 or self._batch_wait_log_counter % 100 == 0:
                        loguru.logger.debug(
                            f"Buffer has {len(self._sample_queue)} samples, "
                            f"waiting for {global_batch_size}... "
                            f"(checked {self._batch_wait_log_counter} times)"
                        )
                    return []

                batch_items = []
                # Efficient O(batch_size) implementation using deque's O(1) popleft
                while self._sample_queue:
                    item = self._sample_queue.popleft()
                    batch_items.append(item)
                    if len(batch_items) >= global_batch_size:
                        break
                # Apply length balancing if requested
                if balance_partitions and balance_partitions > 1:
                    batch_refs = self._apply_length_balancing(batch_items, balance_partitions)
                else:
                    batch_refs = [item[1] for item in batch_items]

                # Build cache as list of lists, one for each dp_rank
                self._cache = []
                for rank in range(balance_partitions):
                    self._cache.append(batch_refs[rank * batch_size : (rank + 1) * batch_size])

                self._batch_wait_log_counter = 0  # Reset counter on successful batch
                res = self._cache[dp_rank]
                loguru.logger.info(f"Buffer return {global_batch_size} samples, {len(self._sample_queue)} samples left")
                return res
            # With filter plugin, use O(N) filtering and reconstruction
            else:
                # 1. The filtering process does not consume elements from the queue
                if isinstance(filter_plugin, list):
                    potential_items = []
                    # all_items = [item for item in self._sample_queue]
                    for item in self._sample_queue:
                        if all(filter_func(item[0]) for filter_func in filter_plugin):
                            potential_items.append(item)
                else:
                    potential_items = [item for item in self._sample_queue if filter_plugin(item[0])]
                # 2. Check if there are enough samples
                if len(potential_items) < global_batch_size:
                    self._batch_wait_log_counter += 1
                    if self._batch_wait_log_counter == 1 or self._batch_wait_log_counter % 100 == 0:
                        loguru.logger.debug(
                            f"Buffer has {len(potential_items)} samples, "
                            f"waiting for {global_batch_size}... "
                            f"(checked {self._batch_wait_log_counter} times)"
                        )
                    return []
                potential_items = potential_items[:global_batch_size]
                # 4. Efficiently remove the selected items from the original queue
                # Use ObjectRef (guaranteed unique and hashable) to identify items for removal
                refs_to_remove = {item[1] for item in potential_items}
                self._sample_queue = deque(item for item in self._sample_queue if item[1] not in refs_to_remove)
                # Apply length balancing if requested
                if balance_partitions and balance_partitions > 1:
                    batch_refs = self._apply_length_balancing(potential_items, balance_partitions)
                else:
                    batch_refs = [item[1] for item in potential_items]
                for rank in range(balance_partitions):
                    self._cache.append(batch_refs[rank * batch_size : (rank + 1) * batch_size])
                self._batch_wait_log_counter = 0  # Reset counter on successful batch
                res = self._cache[dp_rank]

                return res

    def _apply_length_balancing(
        self,
        batch_items: list[tuple[SampleInfo, ray.ObjectRef]],
        k_partitions: int,
        keep_mini_batch=False,
    ) -> list[ray.ObjectRef]:
        """Applies the length balancing algorithm to reorder samples.
        Uses the LPT (Longest Processing Time) algorithm to reorder samples so that
        if they are evenly distributed among k_partitions workers, the sum of
        sample lengths for each worker is as balanced as possible.

        Supports Group N: samples with the same uid will be assigned to the same partition,
        ensuring correct group-relative advantage computation for GRPO and similar algorithms.

        Args:
            batch_items: A list of (SampleInfo, ObjectRef) tuples.
            k_partitions: The number of partitions (typically the DP size).
            keep_mini_batch: Whether to keep mini-batch structure during balancing.

        Returns:
            A reordered list of ObjectRefs.
        """
        # ========== Step 1: Group samples by uid ==========
        uid_to_indices = defaultdict(list)
        for idx, (sample_info, _) in enumerate(batch_items):
            uid = sample_info.uid if sample_info.uid is not None else str(idx)
            uid_to_indices[uid].append(idx)

        # Check if grouping is needed (max_group_size > 1 means we have Group N)
        max_group_size = max(len(indices) for indices in uid_to_indices.values()) if uid_to_indices else 1

        if max_group_size == 1:
            # No grouping needed, use original single-sample balancing logic
            return self._apply_length_balancing_single_sample(batch_items, k_partitions, keep_mini_batch)

        # ========== Step 2: Calculate workload for each Group ==========
        group_list = list(uid_to_indices.keys())  # All unique uids
        group_workloads = []
        for uid in group_list:
            indices = uid_to_indices[uid]
            # Group workload = sum of all samples' sum_tokens in the group
            total_tokens = sum(batch_items[i][0].sum_tokens for i in indices)
            group_workloads.append(total_tokens)

        # ========== Step 3: Balance Groups across partitions ==========
        workload_lst = calculate_workload(group_workloads)

        # Check if number of groups is divisible by k_partitions
        num_groups = len(group_list)
        if num_groups < k_partitions:
            loguru.logger.warning(
                f"Number of groups ({num_groups}) is less than partitions ({k_partitions}). "
                f"Some partitions will be empty. Falling back to single-sample balancing."
            )
            return self._apply_length_balancing_single_sample(batch_items, k_partitions, keep_mini_batch)

        equal_size = num_groups % k_partitions == 0
        if not equal_size:
            loguru.logger.warning(
                f"Number of groups ({num_groups}) is not divisible by partitions ({k_partitions}). "
                f"Some partitions may have uneven group counts."
            )

        # Partition groups across workers
        group_partitions = get_seqlen_balanced_partitions(workload_lst, k_partitions=k_partitions, equal_size=equal_size)

        # ========== Step 4: Expand groups to samples, keeping group integrity ==========
        reordered_refs = []
        for partition_group_indices in group_partitions:
            for group_idx in partition_group_indices:
                uid = group_list[group_idx]
                sample_indices = uid_to_indices[uid]
                # Add all samples of the same group together, preserving original order within group
                for sample_idx in sample_indices:
                    reordered_refs.append(batch_items[sample_idx][1])

        loguru.logger.debug(
            f"Applied GROUP-aware length balancing: "
            f"{len(batch_items)} samples in {num_groups} groups (group_size={max_group_size}) "
            f"reordered into {k_partitions} partitions"
        )

        return reordered_refs

    def _apply_length_balancing_single_sample(
        self,
        batch_items: list[tuple[SampleInfo, ray.ObjectRef]],
        k_partitions: int,
        keep_mini_batch=False,
    ) -> list[ray.ObjectRef]:
        """
        This is used when there's no Group N (each uid has only one sample).

        Args:
            batch_items: A list of (SampleInfo, ObjectRef) tuples.
            k_partitions: The number of partitions (typically the DP size).
            keep_mini_batch: Whether to keep mini-batch structure during balancing.

        Returns:
            A reordered list of ObjectRefs.
        """
        # Extract the length of each sample.
        # Use sum_tokens as the length metric (includes prompt + response).
        seqlen_list = [item[0].sum_tokens for item in batch_items]

        # Use the karmarkar_karp balance
        workload_lst = calculate_workload(seqlen_list)
        # Decouple the DP balancing and mini-batching.
        if keep_mini_batch:
            minibatch_size = self.ppo_mini_batch_size
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(self.world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=self.world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=self.world_size, equal_size=True)

        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition

        # Reorder the samples based on the partitioning result.
        # Concatenate the partitions in order: [all samples from partition_0, all from partition_1, ...]
        reordered_refs = []
        for partition in global_partition_lst:
            for original_idx in partition:
                reordered_refs.append(batch_items[original_idx][1])

        loguru.logger.debug(f"Applied length balancing: {len(batch_items)} samples reordered into {k_partitions} partitions")

        return reordered_refs

    async def get_all_by_filter(self, filter_plugin: Callable[[SampleInfo], bool]) -> list[ray.ObjectRef]:
        """
        Gets ALL sample ObjectRefs that match the filter plugin, consuming them from the queue.
        This is useful for pipeline-based data passing where a downstream stage needs the
        entire output of an upstream stage.
        """
        async with self.lock:
            # 1. Find all items that match the filter.
            items_to_return = [item for item in self._sample_queue if filter_plugin(item[0])]

            if not items_to_return:
                return []

            # 2. Extract their ObjectRefs.
            batch_refs = [item[1] for item in items_to_return]

            # 3. Efficiently remove the selected items from the original queue.
            refs_to_remove = {ref for ref in batch_refs}
            self._sample_queue = deque(item for item in self._sample_queue if item[1] not in refs_to_remove)

            return batch_refs

    async def get_valid_size(self) -> int:
        """Returns the number of samples in the current queue."""
        async with self.lock:
            return len(self._sample_queue)

    async def peek_source_dp_size(self, filter_plugin: Callable[[SampleInfo], bool]) -> int | None:
        """
        Peek at the source_dp_size of matching samples without consuming them.

        Args:
            filter_plugin: Filter function to find matching samples

        Returns:
            The source_dp_size if found, None otherwise
        """
        async with self.lock:
            for sample_info, _ in self._sample_queue:
                if filter_plugin(sample_info):
                    source_dp_size = sample_info.dict_info.get("source_dp_size")
                    if source_dp_size is not None:
                        return source_dp_size
            return None

    # TODO: supporty for async train
    def reset_cache(self):
        loguru.logger.warning("reset datacoordinator")
        self._sample_queue.clear()
        self._cache = []

    def clear_cache(self):
        loguru.logger.warning(f"clear cache of datacoordinator, {len(self._sample_queue)} left")
        self._cache = []

    def __repr__(self) -> str:
        return f"<DataCoordinator(total_samples={len(self._sample_queue)})>"

    # # dataloader function
    @ray.method(concurrency_group="dataloader")
    def init_dataloader(self, config: SiiRLArguments):
        # Not set async factor here since we control in rolloutManager prefetch_thread
        self.dataloader = DataLoaderNode(
            global_config=config,
            config={
                "group_world_size": 1,
                "group_rank": 0,
                "group_parallel_size": 1,
                "num_loader_workers": config.data.num_loader_workers,
                "auto_repeat": config.data.auto_repeat,
            },
        )

    @ray.method(concurrency_group="dataloader")
    def epoch_info(self):
        return self.dataloader.total_training_steps, self.dataloader.num_train_batches

    @ray.method(concurrency_group="dataloader")
    def val_info(self):
        return self.dataloader.num_val_batches, self.dataloader.val_batch_size

    @ray.method(concurrency_group="dataloader")
    async def prepare_data(self, train_batch_size):
        """
        Start or wake up background moving task.
        Keep at least ``train_batch_size`` samples in pending_queue.
        """
        reserve = max(0, int(train_batch_size))
        self._prepare_data_reserve = reserve
        self._prepare_data_stop = False
        if self._prepare_data_task is None or self._prepare_data_task.done():
            self._prepare_data_task = asyncio.create_task(self._prepare_data_loop())
        # Wake up the background task.
        self._prepare_data_event.set()
        return True

    async def _prepare_data_loop(self):
        while not self._prepare_data_stop:
            await self._prepare_data_event.wait()
            self._prepare_data_event.clear()

            while not self._prepare_data_stop:
                moved = False
                reserve = self._prepare_data_reserve
                moved_count = 0
                while self.pending_queue.qsize() > reserve:
                    sample = self.pending_queue.get_nowait()
                    await self.dataloader_queue.put(sample)
                    moved = True
                    moved_count += 1
                    if moved_count % 128 == 0:
                        await asyncio.sleep(0)

                if not moved:
                    break
                await asyncio.sleep(0)

    @ray.method(concurrency_group="dataloader")
    async def run_dataloader_single_sample(self, is_validate=False):
        try:
            batch = self.dataloader.run_single_sample(is_validation_step=is_validate)
        except StopIteration:
            return False
        if batch is None:
            return False
        uid_base = 0
        if not is_validate:
            uid_base = self._next_train_uid
            self._next_train_uid += len(batch["input_ids"])
        tensor_dict = preprocess_dataloader(batch, uid_base=uid_base)
        samples = await Dict2Samples(tensor_dict, True)
        if is_validate:
            for sample in samples:
                await self.dataloader_val_queue.put(sample)
        else:
            for sample in samples:
                # Expand each prompt into rollout_n replicas dispatched independently
                # (so long-tail prompts can't stall on one worker). Each replica is
                # tagged with its slot index so DataCoordinator.put can write it back
                # into the pre-created SampleGroup at _pending_groups[uid].
                uid = str(sample.uid)
                self._pending_groups[uid] = SampleGroup(uid=uid, rollout_n=self.rollout_n)
                for i in range(self.rollout_n):
                    replica = sample if i == 0 else copy.deepcopy(sample)
                    replica.replica_index = i
                    await self.pending_queue.put(replica)
            self._prepare_data_event.set()
        return True

    @ray.method(concurrency_group="dataloader")
    async def get_dataloader_size(self, train_batch_size):
        data_queue = self.dataloader_queue
        remain_pending_size = self.pending_queue.qsize() - train_batch_size
        return data_queue.qsize() + (remain_pending_size if remain_pending_size > 0 else 0)

    @ray.method(concurrency_group="dataloader")
    async def run_dataloader(self, epoch=0, is_validate=False):
        try:
            batch = self.dataloader.run(epoch, is_validation_step=is_validate)
        except StopIteration:
            return False
        if batch is None:
            return False
        uid_base = 0
        # if not is_validate:
        #     uid_base = self._next_train_uid
        #     self._next_train_uid += len(batch["input_ids"])
        tensor_dict = preprocess_dataloader(batch, uid_base=uid_base)
        samples = await Dict2Samples(tensor_dict, True)
        if is_validate:
            for sample in samples:
                await self.dataloader_val_queue.put(sample)
        else:
            for sample in samples:
                # Expand each prompt into rollout_n replicas dispatched independently
                # (so long-tail prompts can't stall on one worker). Each replica is
                # tagged with its slot index so DataCoordinator.put can write it back
                # into the pre-created SampleGroup at _pending_groups[uid].
                uid = str(sample.uid)
                self._pending_groups[uid] = SampleGroup(uid=uid, rollout_n=self.rollout_n)
                for i in range(self.rollout_n):
                    replica = sample if i == 0 else copy.deepcopy(sample)
                    replica.replica_index = i
                    await self.pending_queue.put(replica)
            self._prepare_data_event.set()
        return True

    @ray.method(concurrency_group="dataloader")
    async def get_dataloader(self, batch_size, is_validate=False):
        # TODO: current logic is only fetch the left sample in dataloader queue, we can optimize by cache-aware prefetch in rollout manager
        data_queue = self.dataloader_queue
        if is_validate:
            data_queue = self.dataloader_val_queue
        if data_queue.qsize() > batch_size:
            return [await data_queue.get() for _ in range(batch_size)]
        else:
            all_popped = []
            for _ in range(data_queue.qsize()):
                all_popped.append(await data_queue.get())
            return all_popped

    @ray.method(concurrency_group="dataloader")
    async def put_partial(self, sample) -> None:
        """Push a partial-rollout sample (aborted by weight sync) into the shared
        cancel queue. Any rollout worker may pull it later via ``get_partial``.

        Note: validate-path samples must NOT be pushed here — only train-mode
        partials carry the partial_agent_data needed for resume.
        """
        await self._cancel_queue.put(sample)

    @ray.method(concurrency_group="dataloader")
    async def get_partial(self, batch_size: int) -> list:
        """Drain up to ``batch_size`` partial-rollout samples from the shared
        cancel queue. Returns fewer samples (or an empty list) if the queue is
        shorter than requested. Called by rollout workers in ``get_sample`` to
        prioritize finishing partial work before pulling fresh data.
        """
        if batch_size <= 0:
            return []
        # Mirror get_dataloader: bounded by current qsize so we never block waiting
        # for partial samples that may never arrive (workers must fall through to
        # fresh data instead). Safe under concurrency_group="dataloader" serialization.
        if self._cancel_queue.qsize() > batch_size:
            return [await self._cancel_queue.get() for _ in range(batch_size)]
        else:
            all_popped = []
            for _ in range(self._cancel_queue.qsize()):
                all_popped.append(await self._cancel_queue.get())
            return all_popped

    @ray.method(concurrency_group="dataloader")
    async def get_partial_size(self) -> int:
        """Return the current size of the shared cancel queue (for monitoring)."""
        return self._cancel_queue.qsize()

    @ray.method(concurrency_group="dataloader")
    def save_dataloader_state(self):
        """Save dataloader state and pending dataloader queues."""
        if self.dataloader is None:
            return None
        # Dereference ObjectRefs in _pending_groups so the snapshot is usable
        # after a cluster restart (ObjectRef IDs do not survive re-init).
        pending_groups = {}
        for uid, group in self._pending_groups.items():
            slots = []
            for slot in group.replicas:
                if slot is None:
                    slots.append(None)
                else:
                    info, ref = slot
                    slots.append((info, ray.get(ref)))
            pending_groups[uid] = {
                "uid": group.uid,
                "rollout_n": group.rollout_n,
                "replicas": slots,
            }
        return {
            "dataloader_state": self.dataloader.state_dict(),
            "pending_queue": list(self.pending_queue._queue),
            "train_queue": list(self.dataloader_queue._queue),
            "val_queue": list(self.dataloader_val_queue._queue),
            "next_train_uid": self._next_train_uid,
            "pending_groups": pending_groups,
        }

    @ray.method(concurrency_group="dataloader")
    def load_dataloader_state(self, state_dict):
        """Load dataloader state and pending dataloader queues."""
        if self.dataloader is None or state_dict is None:
            return
        if isinstance(state_dict, dict) and "dataloader_state" in state_dict:
            self.dataloader.load_state_dict(state_dict["dataloader_state"])
            self.pending_queue: asyncio.Queue = asyncio.Queue()
            pending_items = list(state_dict.get("pending_queue", []))
            for item in pending_items:
                self.pending_queue.put_nowait(item)
            self.dataloader_queue: asyncio.Queue = asyncio.Queue()
            train_items = list(state_dict.get("train_queue", []))
            for item in train_items:
                self.dataloader_queue.put_nowait(item)
            self.dataloader_val_queue = asyncio.Queue()
            val_items = list(state_dict.get("val_queue", []))
            for item in val_items:
                self.dataloader_val_queue.put_nowait(item)
            if "next_train_uid" in state_dict:
                self._next_train_uid = int(state_dict["next_train_uid"])
            else:
                queued_uids = [int(sample.uid) for sample in train_items if getattr(sample, "uid", None) is not None]
                self._next_train_uid = (max(queued_uids) + 1) if queued_uids else 0
            # Rehydrate _pending_groups: re-ray.put the Samples to get fresh ObjectRefs
            # valid in the current cluster session.
            self._pending_groups = {}
            for uid, group_state in (state_dict.get("pending_groups") or {}).items():
                group = SampleGroup(uid=group_state["uid"], rollout_n=int(group_state["rollout_n"]))
                for i, slot in enumerate(group_state["replicas"]):
                    if slot is None:
                        continue
                    info, sample = slot
                    group.replicas[i] = (info, ray.put(sample))
                self._pending_groups[uid] = group
            self._prepare_data_event.set()
            return

        self.dataloader.load_state_dict(state_dict)
        self.pending_queue = asyncio.Queue()
        self.dataloader_queue = asyncio.Queue()
        self.dataloader_val_queue = asyncio.Queue()
        self._pending_groups = {}
        self._next_train_uid = 0


# ====================================================================
# Initialization Logic
# ====================================================================


def init_data_coordinator(num_buffers: int, ppo_mini_batch_size: int, world_size: int, rollout_n: int) -> ray.actor.ActorHandle:
    """
    Initializes the data coordination system, which includes a global DataCoordinator
    and multiple distributed DataBuffers. Returns a single, unified DataCoordinator
    handle to the user.

    Args:
        num_buffers: The number of distributed DataBuffer instances to create,
                     usually equal to the number of nodes or total GPUs.
        force_local: If True, forces all Buffers to be created on the local node,
                     for single-machine testing.

    Returns:
        The Actor handle for the DataCoordinator.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before calling init_data_coordinator.")

    # 1. Create or get the globally unique DataCoordinator
    # Use a global name to ensure the coordinator's uniqueness
    coordinator_name = "global_data_coordinator"
    try:
        coordinator = ray.get_actor(coordinator_name)
        loguru.logger.info(f"Connected to existing DataCoordinator actor '{coordinator_name}'.")
    except ValueError:
        loguru.logger.info(f"Creating new DataCoordinator actor with global name '{coordinator_name}'.")
        coordinator = DataCoordinator.options(name=coordinator_name, lifetime="detached").remote(
            nnodes=num_buffers,
            ppo_mini_batch_size=ppo_mini_batch_size,
            world_size=world_size,
            rollout_n=rollout_n,
        )

    return coordinator
