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

"""Shared constants for validate-reuse coordination and worker pool lifecycle."""

# === Sync protocol ===
SYNC_TIMEOUT_S = 120
SYNC_LOG_INTERVAL_S = 5.0
GATE_POLL_INTERVAL_MS = 50
SYNC_RETRY_SLEEP_S = 0.05
# Session id sentinel used before rank0 allocates a real sync session.
NO_SESSION_ID = -1

# === Worker pool lifecycle ===
GRACEFUL_SHUTDOWN_TIMEOUT_S = 15
RECREATE_COOLDOWN_S = 3.0
# Rotate port windows across sessions to reduce TIME_WAIT collisions.
PORT_STRIDE = 128
PORT_CYCLE = 10
PORT_RETRY_SLOTS = 3

# === Validation progress ===
PROGRESS_POLL_INTERVAL_S = 2.0

# === Trainer-side state tensor ===
# Fixed indices keep broadcast payload layout stable across trainer ranks.
STATE_ACTIVE_IDX = 0
STATE_SYNC_REQUIRED_IDX = 1
STATE_SESSION_ID_IDX = 2
STATE_NUM_FIELDS = 3
