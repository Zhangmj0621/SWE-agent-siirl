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
Unit tests for off-policy data filtering functionality in DataCoordinator.
"""

import asyncio
import unittest

import ray
import torch
from tensordict import TensorDict

from siirl.data_coordinator.data_buffer import init_data_coordinator
from siirl.data_coordinator.sample import SampleInfo


class TestOffPolicyFiltering(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the off-policy data filtering feature in DataCoordinator.

    Tests cover:
    - min_version filtering: discarding stale samples
    - FIFO ordering: samples above min_version are returned in order
    - Edge cases: empty queue, all samples stale, etc.
    """

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True, logging_level="error")

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        """Create a new, clean DataCoordinator for each test."""
        self.coordinator = init_data_coordinator(num_buffers=1, ppo_mini_batch_size=1, world_size=1, rollout_n=1)
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 0)

    async def asyncTearDown(self):
        """Destroy the actor after each test."""
        ray.kill(self.coordinator, no_restart=True)
        await asyncio.sleep(0.1)

    def _create_mock_sample(self, content_id: int) -> TensorDict:
        """Helper to create a sample with identifiable content."""
        return TensorDict({"data": torch.tensor([[content_id]])}, batch_size=[1])

    def _create_sample_info(self, tokens: int, weight_version: int, uid: str = None) -> SampleInfo:
        """Helper to create a SampleInfo object with weight_version.

        ``replica_index=0`` together with coordinator ``rollout_n=1`` makes each
        ``put`` act as an immediate single-slot group that flushes to
        ``_sample_queue`` — matches the pre-refactor append-on-put behavior these
        tests rely on.
        """
        return SampleInfo(
            sum_tokens=tokens,
            prompt_length=tokens,
            response_length=0,
            weight_version=weight_version,
            uid=uid or str(tokens),
            replica_index=0,
        )

    # === Test Cases for min_version filtering ===

    async def test_min_version_discards_stale_samples(self):
        """Test that samples with version < min_version are discarded."""
        # Put samples with different versions
        versions = [1, 2, 3, 4, 5]
        for v in versions:
            sample_ref = ray.put(self._create_mock_sample(v * 100))
            sample_info = self._create_sample_info(tokens=128, weight_version=v)
            await self.coordinator.put.remote(sample_info, sample_ref)

        self.assertEqual(await self.coordinator.get_valid_size.remote(), 5)

        # Request batch with min_version=3 (should discard versions 1, 2)
        batch_refs = await self.coordinator.get_batch.remote(
            batch_size=3,
            dp_rank=0,
            balance_partitions=1,
            min_version=3,
        )

        # Should get 3 samples (versions 3, 4, 5)
        self.assertEqual(len(batch_refs), 3)

        # Clear cache for next request
        await self.coordinator.clear_cache.remote()

        # Queue should be empty now (3 returned, 2 discarded)
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 0)

        # Verify returned data
        retrieved_data = ray.get(batch_refs)
        retrieved_ids = sorted([d.get("data").item() for d in retrieved_data])
        self.assertEqual(retrieved_ids, [300, 400, 500])

    async def test_min_version_keeps_valid_samples_in_fifo_order(self):
        """Test that samples >= min_version are returned in FIFO order."""
        # Put samples with versions in specific order
        order_data = [(3, 300), (5, 500), (4, 400), (3, 301), (6, 600)]
        for version, content_id in order_data:
            sample_ref = ray.put(self._create_mock_sample(content_id))
            sample_info = self._create_sample_info(tokens=128, weight_version=version, uid=str(content_id))
            await self.coordinator.put.remote(sample_info, sample_ref)

        # Request batch with min_version=4
        batch_refs = await self.coordinator.get_batch.remote(
            batch_size=3,
            dp_rank=0,
            balance_partitions=1,
            min_version=4,
        )

        # Should get samples with version >= 4: (5,500), (4,400), (6,600) in FIFO order
        self.assertEqual(len(batch_refs), 3)

        await self.coordinator.clear_cache.remote()

        retrieved_data = ray.get(batch_refs)
        retrieved_ids = [d.get("data").item() for d in retrieved_data]
        # FIFO order should be: 500, 400, 600
        self.assertEqual(retrieved_ids, [500, 400, 600])

    async def test_min_version_all_samples_stale(self):
        """Test behavior when all samples are below min_version."""
        # Put samples with old versions
        for v in [1, 2, 3]:
            sample_ref = ray.put(self._create_mock_sample(v * 100))
            sample_info = self._create_sample_info(tokens=128, weight_version=v)
            await self.coordinator.put.remote(sample_info, sample_ref)

        self.assertEqual(await self.coordinator.get_valid_size.remote(), 3)

        # Request batch with min_version=5 (all samples are stale)
        batch_refs = await self.coordinator.get_batch.remote(
            batch_size=2,
            dp_rank=0,
            balance_partitions=1,
            min_version=5,
        )

        # Should return empty (not enough valid samples after filtering)
        self.assertEqual(len(batch_refs), 0)

        # All stale samples should be discarded
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 0)

    async def test_min_version_zero_accepts_all(self):
        """Test that min_version=0 accepts all samples (including version 0)."""
        # Put samples with version 0 and above
        versions = [0, 1, 2]
        for v in versions:
            sample_ref = ray.put(self._create_mock_sample(v * 100))
            sample_info = self._create_sample_info(tokens=128, weight_version=v)
            await self.coordinator.put.remote(sample_info, sample_ref)

        batch_refs = await self.coordinator.get_batch.remote(
            batch_size=3,
            dp_rank=0,
            balance_partitions=1,
            min_version=0,
        )

        self.assertEqual(len(batch_refs), 3)
        await self.coordinator.clear_cache.remote()

        retrieved_data = ray.get(batch_refs)
        retrieved_ids = sorted([d.get("data").item() for d in retrieved_data])
        self.assertEqual(retrieved_ids, [0, 100, 200])

    async def test_min_version_none_no_filtering(self):
        """Test that min_version=None skips version filtering entirely."""
        # Put samples with different versions
        versions = [1, 2, 3]
        for v in versions:
            sample_ref = ray.put(self._create_mock_sample(v * 100))
            sample_info = self._create_sample_info(tokens=128, weight_version=v)
            await self.coordinator.put.remote(sample_info, sample_ref)

        # Request without min_version (should keep all)
        batch_refs = await self.coordinator.get_batch.remote(
            batch_size=3,
            dp_rank=0,
            balance_partitions=1,
            min_version=None,  # No filtering
        )

        self.assertEqual(len(batch_refs), 3)
        await self.coordinator.clear_cache.remote()
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 0)

    async def test_min_version_partial_batch(self):
        """Test when valid samples exist but not enough for full batch."""
        # Put 5 samples: versions [1, 2, 3, 4, 5]
        for v in range(1, 6):
            sample_ref = ray.put(self._create_mock_sample(v * 100))
            sample_info = self._create_sample_info(tokens=128, weight_version=v)
            await self.coordinator.put.remote(sample_info, sample_ref)

        # Request batch of 5 with min_version=4 (only 2 valid samples)
        batch_refs = await self.coordinator.get_batch.remote(
            batch_size=5,
            dp_rank=0,
            balance_partitions=1,
            min_version=4,
        )

        # Should return empty because we don't have enough valid samples
        self.assertEqual(len(batch_refs), 0)

        # Stale samples (v1, v2, v3) should be discarded, valid ones (v4, v5) kept
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 2)

    async def test_min_version_exact_boundary(self):
        """Test samples exactly at min_version boundary are included."""
        # Put samples at boundary version
        for v in [5, 5, 5]:
            sample_ref = ray.put(self._create_mock_sample(500 + v))
            sample_info = self._create_sample_info(tokens=128, weight_version=v, uid=f"v{v}_{500+v}")
            await self.coordinator.put.remote(sample_info, sample_ref)

        # min_version=5 should include all version=5 samples
        batch_refs = await self.coordinator.get_batch.remote(
            batch_size=3,
            dp_rank=0,
            balance_partitions=1,
            min_version=5,
        )

        self.assertEqual(len(batch_refs), 3)
        await self.coordinator.clear_cache.remote()

        retrieved_data = ray.get(batch_refs)
        for d in retrieved_data:
            self.assertEqual(d.get("data").item(), 505)


class TestOffPolicyWithMultiplePartitions(unittest.IsolatedAsyncioTestCase):
    """
    Test off-policy filtering with balance_partitions > 1.
    """

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True, logging_level="error")

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    async def asyncSetUp(self):
        self.coordinator = init_data_coordinator(num_buffers=1, ppo_mini_batch_size=2, world_size=2, rollout_n=1)
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 0)

    async def asyncTearDown(self):
        ray.kill(self.coordinator, no_restart=True)
        await asyncio.sleep(0.1)

    def _create_mock_sample(self, content_id: int) -> TensorDict:
        return TensorDict({"data": torch.tensor([[content_id]])}, batch_size=[1])

    def _create_sample_info(self, tokens: int, weight_version: int, uid: str = None) -> SampleInfo:
        return SampleInfo(
            sum_tokens=tokens,
            prompt_length=tokens,
            response_length=0,
            weight_version=weight_version,
            uid=uid or str(tokens),
            replica_index=0,
        )

    async def test_min_version_with_partitions(self):
        """Test min_version filtering works correctly with multiple partitions."""
        # Put 8 samples: versions [1,1,2,2,3,3,4,4]
        for v in [1, 1, 2, 2, 3, 3, 4, 4]:
            sample_ref = ray.put(self._create_mock_sample(v * 100))
            sample_info = self._create_sample_info(tokens=128, weight_version=v, uid=f"v{v}_{v*100}")
            await self.coordinator.put.remote(sample_info, sample_ref)

        self.assertEqual(await self.coordinator.get_valid_size.remote(), 8)

        # Request with batch_size=2, balance_partitions=2
        # Total global batch = 2 * 2 = 4
        # min_version=3 should discard versions 1, 2 (4 samples)
        # Remaining valid: versions 3, 4 (4 samples) - exactly enough

        # dp_rank=0 gets first half
        batch_refs_0 = await self.coordinator.get_batch.remote(
            batch_size=2,
            dp_rank=0,
            balance_partitions=2,
            min_version=3,
        )

        self.assertEqual(len(batch_refs_0), 2)

        # dp_rank=1 gets second half (from cache)
        batch_refs_1 = await self.coordinator.get_batch.remote(
            batch_size=2,
            dp_rank=1,
            balance_partitions=2,
            min_version=3,
        )

        self.assertEqual(len(batch_refs_1), 2)

        await self.coordinator.clear_cache.remote()

        # All 4 valid samples consumed, 4 stale discarded
        self.assertEqual(await self.coordinator.get_valid_size.remote(), 0)


if __name__ == "__main__":
    unittest.main()
