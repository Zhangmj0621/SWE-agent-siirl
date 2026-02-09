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

"""State coordinator for validate-reuse sync sessions."""

import asyncio
import time


class ValidateReuseCoordinator:
    """Owns validate-reuse sync session state and transition logic."""

    def __init__(self, trainer_world_size: int):
        self._trainer_world_size = max(0, int(trainer_world_size))
        self._session_counter = 0
        self._sync_required = False
        self._phase = "IDLE"
        self._abort_reason = ""
        self._active_session_id: int | None = None
        self._begin_ranks: set[int] = set()
        self._synced_ranks: set[int] = set()
        self._sync_plan_logged_ranks: set[int] = set()

    @property
    def trainer_world_size(self) -> int:
        return self._trainer_world_size

    @property
    def sync_required(self) -> bool:
        return self._sync_required

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def abort_reason(self) -> str:
        return self._abort_reason

    @property
    def active_session_id(self) -> int | None:
        return self._active_session_id

    def synced_count(self) -> int:
        return len(self._synced_ranks)

    def missing_ranks(self) -> list[int]:
        return sorted(set(range(self._trainer_world_size)) - self._synced_ranks)

    def clear_sync_plan_logs(self) -> None:
        self._sync_plan_logged_ranks.clear()

    def should_log_sync_plan(self, trainer_rank: int) -> bool:
        if trainer_rank in self._sync_plan_logged_ranks:
            return False
        self._sync_plan_logged_ranks.add(trainer_rank)
        return True

    def reset(self) -> None:
        self._sync_required = False
        self._phase = "IDLE"
        self._abort_reason = ""
        self._active_session_id = None
        self._begin_ranks.clear()
        self._synced_ranks.clear()
        self._sync_plan_logged_ranks.clear()

    def prepare_for_validation(self, *, sync_required: bool) -> None:
        self.reset()
        self._sync_required = bool(sync_required)

    def get_active_state(self, *, validate_active: bool, global_steps: int, no_session_id: int) -> dict:
        return {
            "active": bool(validate_active),
            "sync_required": bool(self._sync_required),
            "phase": self._phase,
            "session_id": int(self._active_session_id) if self._active_session_id is not None else no_session_id,
            "global_steps": int(global_steps),
        }

    def start_session(self, *, no_session_id: int) -> int:
        if not self._sync_required:
            return no_session_id
        if self._active_session_id is None:
            self._session_counter += 1
            self._active_session_id = self._session_counter
            self._phase = "COLLECTING"
            self._abort_reason = ""
            self._begin_ranks.clear()
        return int(self._active_session_id)

    def mark_begin(self, trainer_rank: int, session_id: int) -> dict:
        if not self._sync_required:
            return {"accepted": False, "reason": "sync_not_required", "phase": self._phase}
        if session_id != self._active_session_id:
            return {"accepted": False, "reason": "stale_session", "phase": self._phase}
        if self._phase == "ABORTED":
            return {"accepted": False, "reason": self._abort_reason, "phase": self._phase}

        self._begin_ranks.add(trainer_rank)
        # RUNNING means every trainer rank has entered the sync session.
        if len(self._begin_ranks) >= self._trainer_world_size > 0:
            self._phase = "RUNNING"
        return {
            "accepted": True,
            "phase": self._phase,
            "begun": len(self._begin_ranks),
            "world_size": self._trainer_world_size,
        }

    def get_gate(self, session_id: int) -> dict:
        if not self._sync_required:
            return {"proceed": False, "phase": "IDLE", "reason": "sync_not_required"}
        if session_id != self._active_session_id:
            return {"proceed": False, "phase": self._phase, "reason": "stale_session"}
        if self._phase == "ABORTED":
            return {"proceed": False, "phase": self._phase, "reason": self._abort_reason or "aborted"}
        if len(self._begin_ranks) >= self._trainer_world_size > 0:
            self._phase = "RUNNING"
            return {"proceed": True, "phase": self._phase}
        return {"proceed": None, "phase": self._phase}

    def abort_session(self, session_id: int, reason: str) -> dict:
        if not self._sync_required:
            return {"aborted": False, "phase": "IDLE"}
        if session_id != self._active_session_id:
            return {"aborted": False, "phase": self._phase, "reason": "stale_session"}
        self._phase = "ABORTED"
        self._abort_reason = reason
        return {"aborted": True, "phase": self._phase, "reason": reason}

    def mark_synced(self, trainer_rank: int, session_id: int | None = None) -> dict:
        if not self._sync_required:
            return {"accepted": False, "reason": "sync_not_required"}
        if session_id is not None and session_id != self._active_session_id:
            return {"accepted": False, "reason": "stale_session"}
        if self._phase == "ABORTED":
            return {"accepted": False, "reason": self._abort_reason}

        self._synced_ranks.add(trainer_rank)
        synced = len(self._synced_ranks)
        missing = self.missing_ranks()
        # Done session clears sync_required so trainers can proceed with train_step.
        if synced >= self._trainer_world_size > 0:
            self._phase = "DONE"
            self._sync_required = False
        return {
            "accepted": True,
            "phase": self._phase,
            "synced": synced,
            "world_size": self._trainer_world_size,
            "missing": missing,
        }

    async def wait_synced(
        self,
        *,
        timeout_s: int,
        log_interval_s: float,
        poll_interval_s: float = 0.05,
    ) -> bool:
        from loguru import logger

        if not self._sync_required:
            return True
        if self._trainer_world_size <= 0:
            return False

        start = time.time()
        next_log_at = start + log_interval_s
        deadline = start + timeout_s
        while time.time() < deadline:
            if self._phase == "ABORTED":
                logger.warning("[RolloutManager] Validate reuse sync aborted " f"reason={self._abort_reason or 'unknown'}")
                return False

            synced = len(self._synced_ranks)
            if synced >= self._trainer_world_size:
                logger.info(
                    "[RolloutManager] Validate reuse sync completed "
                    f"in {time.time() - start:.2f}s synced={synced}/{self._trainer_world_size}"
                )
                return True

            now = time.time()
            if now >= next_log_at:
                logger.info(
                    "[RolloutManager] Waiting validate reuse sync "
                    f"elapsed={now - start:.2f}s synced={synced}/{self._trainer_world_size} "
                    f"missing={self.missing_ranks()}"
                )
                next_log_at = now + log_interval_s
            await asyncio.sleep(poll_interval_s)
        return False
