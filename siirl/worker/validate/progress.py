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

"""Validation progress state and driver-side monitor."""

import sys
import time
from contextlib import suppress

import ray


class ValidateProgressTracker:
    """Stores validation progress state in RolloutManager."""

    def __init__(self):
        self._active = False
        self._total = 0
        self._done = 0
        self._workers_active = 0
        self._workers_total = 0
        self._step = 0
        self._rollout_index = 0
        self._last_update_ts = 0.0

    def set_state(
        self,
        *,
        active: bool,
        total: int,
        done: int,
        workers_active: int,
        workers_total: int,
        step: int,
        rollout_index: int,
    ) -> None:
        self._active = bool(active)
        self._total = max(0, int(total))
        self._done = max(0, int(done))
        self._workers_active = max(0, int(workers_active))
        self._workers_total = max(0, int(workers_total))
        self._step = int(step)
        self._rollout_index = int(rollout_index)
        self._last_update_ts = time.time()

    def snapshot(self) -> dict:
        return {
            "active": self._active,
            "total": self._total,
            "done": self._done,
            "workers_active": self._workers_active,
            "workers_total": self._workers_total,
            "step": self._step,
            "rollout_index": self._rollout_index,
            "last_update_ts": self._last_update_ts,
        }


class ValidateProgressMonitor:
    """Driver-side monitor for validation progress."""

    def __init__(self, rollout_manager_name: str):
        self.rollout_manager_name = rollout_manager_name
        self.rollout_manager = None
        self.progress_bar = None
        self.last_key = None
        self.last_progress_rollout_index = -1
        self.last_completed_rollout_index = -1
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None
        self.tqdm = tqdm
        self.use_tqdm = self.tqdm is not None and bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._text_progress_interval_s = 5.0
        self._last_text_progress_ts = 0.0
        self._last_text_done = -1
        self._last_text_rollout_index = -1

    def close(self):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def poll_once(self):
        if self.rollout_manager is None:
            with suppress(Exception):
                self.rollout_manager = ray.get_actor(self.rollout_manager_name)
            if self.rollout_manager is None:
                return

        with suppress(Exception):
            snapshot = ray.get(self.rollout_manager.get_validate_progress_snapshot.remote(), timeout=1)
            self._update(snapshot)

    def _update(self, snapshot: dict):
        total = max(0, int(snapshot.get("total", 0)))
        done = max(0, int(snapshot.get("done", 0)))
        active = bool(snapshot.get("active", False))
        step = int(snapshot.get("step", 0))
        rollout_index = int(snapshot.get("rollout_index", 0))
        workers_active = max(0, int(snapshot.get("workers_active", 0)))
        workers_total = max(0, int(snapshot.get("workers_total", 0)))
        key = (active, total, done, step, rollout_index, workers_active, workers_total)
        if key == self.last_key:
            return
        self.last_key = key

        if self.use_tqdm and active and total > 0:
            if self.progress_bar is None or self.progress_bar.total != total or self.last_progress_rollout_index != rollout_index:
                self.close()
                self.progress_bar = self.tqdm(
                    total=total,
                    desc=f"Validate@step{step}",
                    unit="sample",
                    dynamic_ncols=True,
                    leave=False,
                    file=sys.stdout,
                )
                self.last_progress_rollout_index = rollout_index
            self.progress_bar.n = min(done, total)
            self.progress_bar.set_postfix_str(f"active={workers_active}/{workers_total}", refresh=False)
            self.progress_bar.refresh()
            return

        if self.progress_bar is not None:
            if total > 0:
                self.progress_bar.n = total if not active else min(done, total)
                self.progress_bar.refresh()
            self.close()

        if not self.use_tqdm and total > 0:
            if rollout_index != self._last_text_rollout_index:
                self._last_text_rollout_index = rollout_index
                self._last_text_progress_ts = 0.0
                self._last_text_done = -1

            done_clamped = min(done, total)
            pct = 100.0 * done_clamped / total

            if active:
                now = time.time()
                should_print = self._last_text_done < 0
                if done_clamped > self._last_text_done and now - self._last_text_progress_ts >= self._text_progress_interval_s:
                    should_print = True
                if should_print:
                    print(
                        f"Validate@step{step} progress: {done_clamped}/{total} ({pct:.1f}%), "
                        f"workers_active={workers_active}/{workers_total}, rollout_index={rollout_index}",
                        flush=True,
                    )
                    self._last_text_progress_ts = now
                    self._last_text_done = done_clamped
                return

            if rollout_index > self.last_completed_rollout_index:
                print(
                    f"Validate@step{step} done: {done_clamped}/{total} ({pct:.1f}%), "
                    f"workers_active={workers_active}/{workers_total}, rollout_index={rollout_index}",
                    flush=True,
                )
                self.last_completed_rollout_index = rollout_index
                self._last_text_progress_ts = 0.0
                self._last_text_done = -1
