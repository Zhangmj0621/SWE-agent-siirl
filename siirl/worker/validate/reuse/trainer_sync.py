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

"""Trainer-side validate-reuse synchronization protocol."""

import time
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import ray
import torch
import torch.distributed as dist
from loguru import logger

from siirl.utils.distributed_utils import get_gloo_group
from siirl.worker.validate.reuse.constants import (
    GATE_POLL_INTERVAL_MS,
    NO_SESSION_ID,
    STATE_ACTIVE_IDX,
    STATE_NUM_FIELDS,
    STATE_SESSION_ID_IDX,
    STATE_SYNC_REQUIRED_IDX,
)


class ValidateGateDecision(str, Enum):
    PROCEED = "proceed"
    RETRY_SYNC = "retry_sync"


class ValidateReuseTrainerSync:
    """Encapsulates Trainer-side validate-reuse sync flow."""

    def __init__(
        self,
        *,
        rollout_manager: Any,
        rank: int,
        config,
        sync_workers_fn: Callable[[Sequence, Sequence | None, bool], None],
        get_regular_workers_fn: Callable[[], list],
        ensure_regular_workers_fn: Callable[[Sequence], None],
        get_current_weight_version_fn: Callable[[], int],
    ):
        self._rollout_manager = rollout_manager
        self._rank = rank
        self._config = config
        self._sync_workers_fn = sync_workers_fn
        self._get_regular_workers_fn = get_regular_workers_fn
        self._ensure_regular_workers_fn = ensure_regular_workers_fn
        self._get_current_weight_version_fn = get_current_weight_version_fn
        self._wait_session_id = NO_SESSION_ID
        self._wait_started_at = 0.0

    def _rpc_timeout_s(self) -> int:
        return max(1, int(getattr(self._config.trainer, "param_sync_rpc_timeout_s", 120)))

    def _begin_timeout_s(self) -> int:
        return max(1, int(getattr(self._config.trainer, "validate_reuse_begin_timeout_s", 30)))

    @staticmethod
    def _poll_sleep_s() -> float:
        return max(0.001, float(GATE_POLL_INTERVAL_MS) / 1000.0)

    def try_sync(self) -> bool:
        if self._rollout_manager is None:
            return False

        sync_plan = ray.get(self._rollout_manager.get_validate_reuse_sync_plan.remote(self._rank))
        distributed_workers = sync_plan.get("distributed_workers", [])
        tensor_workers = sync_plan.get("tensor_workers", [])
        needs_sync_local = 1 if (distributed_workers or tensor_workers) else 0
        needs_sync_tensor = torch.tensor([needs_sync_local], dtype=torch.int32)
        # Any rank requiring sync forces all ranks onto the same collective path.
        dist.all_reduce(needs_sync_tensor, op=dist.ReduceOp.MAX, group=get_gloo_group())
        if needs_sync_tensor.item() == 0:
            return False

        session_id_tensor = torch.tensor([NO_SESSION_ID], dtype=torch.int64)
        rpc_timeout_s = self._rpc_timeout_s()
        if self._rank == 0:
            try:
                session_id_tensor[0] = int(ray.get(self._rollout_manager.start_validate_reuse_sync_session.remote(), timeout=rpc_timeout_s))
            except Exception as e:
                logger.error(f"[Trainer rank=0] Failed to allocate validate reuse session: {e}")
                session_id_tensor[0] = NO_SESSION_ID
        # Rank0 is the single session allocator to avoid split-brain sessions.
        dist.broadcast(session_id_tensor, src=0, group=get_gloo_group())
        session_id = int(session_id_tensor.item())
        if session_id < 0:
            return False

        begin_result = ray.get(
            self._rollout_manager.mark_validate_reuse_begin.remote(self._rank, session_id),
            timeout=rpc_timeout_s,
        )
        if not begin_result.get("accepted", False):
            logger.warning(
                f"[Trainer rank={self._rank}] Validate reuse begin rejected: "
                f"phase={begin_result.get('phase')} reason={begin_result.get('reason')}"
            )

        gate_decision = torch.tensor([0], dtype=torch.int32)
        gate_poll_s = self._poll_sleep_s()
        if self._rank == 0:
            deadline = time.time() + self._begin_timeout_s()
            gate_reason = ""
            while True:
                gate_state = ray.get(
                    self._rollout_manager.get_validate_reuse_sync_gate.remote(session_id),
                    timeout=rpc_timeout_s,
                )
                proceed = gate_state.get("proceed")
                if proceed is True:
                    gate_decision[0] = 1
                    break
                if proceed is False:
                    gate_reason = gate_state.get("reason", "aborted")
                    gate_decision[0] = -1
                    break
                if time.time() >= deadline:
                    gate_reason = f"validate reuse begin timeout after {self._begin_timeout_s()}s"
                    ray.get(
                        self._rollout_manager.abort_validate_reuse_sync_session.remote(session_id, gate_reason),
                        timeout=rpc_timeout_s,
                    )
                    gate_decision[0] = -1
                    break
                time.sleep(gate_poll_s)
            if gate_decision.item() < 0:
                logger.warning(f"[Trainer rank=0] Validate reuse sync gate aborted: {gate_reason or 'unknown'}")
        dist.broadcast(gate_decision, src=0, group=get_gloo_group())
        if gate_decision.item() < 0:
            return False

        start = time.time()
        logger.info(
            f"[Trainer rank={self._rank}] Validate reuse sync start: "
            f"distributed_workers={len(distributed_workers)} tensor_workers={len(tensor_workers)} "
            f"weight_version={self._get_current_weight_version_fn()}"
        )
        # Reuse sync must not advance weight_version seen by rollout dataloader.
        self._sync_workers_fn(distributed_workers, tensor_workers, False)
        regular_workers = self._get_regular_workers_fn()
        self._ensure_regular_workers_fn(regular_workers)
        ray.get(
            self._rollout_manager.mark_validate_reuse_synced.remote(self._rank, session_id),
            timeout=rpc_timeout_s,
        )
        logger.info(
            f"[Trainer rank={self._rank}] Validate reuse sync done in {time.time() - start:.2f}s "
            f"weight_version={self._get_current_weight_version_fn()}"
        )
        return True

    def wait_idle(self) -> ValidateGateDecision:
        if self._rollout_manager is None:
            return ValidateGateDecision.PROCEED

        rpc_timeout_s = self._rpc_timeout_s()
        poll_s = self._poll_sleep_s()
        while True:
            state_tensor = torch.zeros(STATE_NUM_FIELDS, dtype=torch.int64)
            state_tensor[STATE_SESSION_ID_IDX] = NO_SESSION_ID
            if self._rank == 0:
                try:
                    state = ray.get(self._rollout_manager.get_validate_active_state.remote(), timeout=rpc_timeout_s)
                    state_tensor[STATE_ACTIVE_IDX] = 1 if bool(state.get("active", False)) else 0
                    state_tensor[STATE_SYNC_REQUIRED_IDX] = 1 if bool(state.get("sync_required", False)) else 0
                    state_tensor[STATE_SESSION_ID_IDX] = int(state.get("session_id", NO_SESSION_ID))
                except Exception as e:
                    logger.warning(f"[Trainer rank=0] Failed to query validate active state: {e}")
                    state_tensor[STATE_ACTIVE_IDX] = 0
                    state_tensor[STATE_SYNC_REQUIRED_IDX] = 0
                    state_tensor[STATE_SESSION_ID_IDX] = NO_SESSION_ID

            dist.broadcast(state_tensor, src=0, group=get_gloo_group())
            is_active = bool(state_tensor[STATE_ACTIVE_IDX].item())
            sync_required = bool(state_tensor[STATE_SYNC_REQUIRED_IDX].item())
            session_id = int(state_tensor[STATE_SESSION_ID_IDX].item())
            if not is_active:
                if self._rank == 0 and self._wait_started_at > 0:
                    elapsed_s = time.time() - self._wait_started_at
                    logger.info(f"[Trainer rank=0] Validate wait finished in {elapsed_s:.2f}s " f"session_id={self._wait_session_id}")
                self._wait_session_id = NO_SESSION_ID
                self._wait_started_at = 0.0
                return ValidateGateDecision.PROCEED

            if sync_required:
                # Validate is active and still waiting for trainer-side sync.
                self._wait_session_id = NO_SESSION_ID
                self._wait_started_at = 0.0
                return ValidateGateDecision.RETRY_SYNC

            if self._wait_started_at <= 0 or self._wait_session_id != session_id:
                self._wait_session_id = session_id
                self._wait_started_at = time.time()
                if self._rank == 0:
                    logger.info(f"[Trainer rank=0] Validate wait start session_id={session_id}")
            time.sleep(poll_s)
