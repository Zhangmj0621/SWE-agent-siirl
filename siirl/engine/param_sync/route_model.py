"""Route model and builders for colocated flattened-bucket sync."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from megatron.core import mpu
from ray.actor import ActorHandle


@dataclass(frozen=True)
class LaneKey:
    """Unique identifier for one colocated sync lane."""

    pp_rank: int
    tp_group_idx: int
    lane_idx: int


@dataclass
class LaneRoute:
    """Route information for one lane."""

    lane: LaneKey
    source_ranks: list[int]
    leader_rank: int
    target_worker: ActorHandle
    target_worker_idx: int


@dataclass
class RoutePlan:
    """Complete routing plan for one PP stage."""

    lanes: list[LaneRoute]
    topology_hash: str
    route_epoch: int
    tp_size: int


def _stable_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _get_dp_world_size() -> int:
    try:
        return int(mpu.get_data_parallel_world_size(with_context_parallel=True))
    except TypeError:
        return int(mpu.get_data_parallel_world_size())


def _calc_global_rank(tp_rank: int, dp_rank: int, pp_rank: int, cp_rank: int) -> int:
    """Megatron global-rank order: tp-cp-dp-pp."""

    tp_size = int(mpu.get_tensor_model_parallel_world_size())
    dp_size = _get_dp_world_size()
    cp_size = int(mpu.get_context_parallel_world_size())
    return ((pp_rank * dp_size + dp_rank) * cp_size + cp_rank) * tp_size + tp_rank


def compute_route_plan(
    rollout_topology: dict,
    rollout_workers: list[ActorHandle],
    config_tp_size: int,
    route_epoch: int = 1,
) -> RoutePlan:
    """Build route plan from rollout topology.

    Each lane corresponds to one rollout TP0 endpoint and maps to one trainer
    (dp, cp, pp) replica for TP=1, or one TP window for TP>1.
    """

    pp_rank = int(mpu.get_pipeline_model_parallel_rank())
    tp_size_runtime = int(mpu.get_tensor_model_parallel_world_size())
    tp_size = max(1, int(config_tp_size))
    if tp_size != tp_size_runtime:
        raise RuntimeError(f"route_plan tp_size mismatch: config={tp_size}, runtime={tp_size_runtime}")

    dp_size = _get_dp_world_size()
    cp_size = int(mpu.get_context_parallel_world_size())
    world_size = int(dp_size * cp_size * tp_size * int(mpu.get_pipeline_model_parallel_world_size()))

    tp0_workers = [w for w in rollout_topology["workers"] if w.get("is_tp0")]
    if len(tp0_workers) != len(rollout_workers):
        raise RuntimeError("route_plan topology mismatch: " f"tp0_workers={len(tp0_workers)} rollout_workers={len(rollout_workers)}")

    max_expected_lanes = dp_size * cp_size

    lanes: list[LaneRoute] = []
    for lane_idx, worker_info in enumerate(tp0_workers):
        tp_group_idx = int(worker_info.get("tp_group_idx", lane_idx))

        # Preferred mapping: lane -> (dp, cp) pair for this PP stage.
        if tp_group_idx < max_expected_lanes:
            dp_rank = tp_group_idx // cp_size
            cp_rank = tp_group_idx % cp_size
            source_ranks = [_calc_global_rank(tp_rank=t, dp_rank=dp_rank, pp_rank=pp_rank, cp_rank=cp_rank) for t in range(tp_size)]
        else:
            # Compatibility fallback for topologies that expose global TP groups.
            start_rank = tp_group_idx * tp_size
            source_ranks = list(range(start_rank, start_rank + tp_size))
        for rank in source_ranks:
            if rank < 0 or rank >= world_size:
                raise RuntimeError(f"route_plan produced invalid source rank={rank} world_size={world_size}")

        lanes.append(
            LaneRoute(
                lane=LaneKey(pp_rank=pp_rank, tp_group_idx=tp_group_idx, lane_idx=lane_idx),
                source_ranks=source_ranks,
                leader_rank=source_ranks[0],
                target_worker=rollout_workers[lane_idx],
                target_worker_idx=int(worker_info["worker_idx"]),
            )
        )

    return RoutePlan(
        lanes=lanes,
        topology_hash=rollout_topology.get("topology_hash", _stable_hash(rollout_topology)),
        route_epoch=max(1, int(route_epoch)),
        tp_size=tp_size,
    )


def validate_route_plan(plan: RoutePlan, group_name: str) -> None:
    """Validate route invariants (fail-fast)."""

    if not plan.lanes:
        raise RuntimeError(f"[{group_name}] Route plan has no lanes")

    leaders: set[int] = set()
    source_ranks: set[int] = set()
    target_workers: set[str] = set()
    lane_ids: set[tuple[int, int, int]] = set()

    for lane in plan.lanes:
        lane_id = (lane.lane.pp_rank, lane.lane.tp_group_idx, lane.lane.lane_idx)
        if lane_id in lane_ids:
            raise RuntimeError(f"[{group_name}] Duplicate lane id {lane_id}")
        lane_ids.add(lane_id)

        if not lane.source_ranks:
            raise RuntimeError(f"[{group_name}] Lane {lane_id} has no source ranks")
        if len(lane.source_ranks) != plan.tp_size:
            raise RuntimeError(f"[{group_name}] Lane {lane_id} source count={len(lane.source_ranks)} != tp_size={plan.tp_size}")

        if lane.leader_rank not in lane.source_ranks:
            raise RuntimeError(f"[{group_name}] Lane {lane_id} leader not in source ranks")
        if lane.leader_rank in leaders:
            raise RuntimeError(f"[{group_name}] Duplicate leader rank {lane.leader_rank}")
        leaders.add(lane.leader_rank)

        for rank in lane.source_ranks:
            if rank in source_ranks:
                raise RuntimeError(f"[{group_name}] Source rank {rank} appears in multiple lanes")
            source_ranks.add(rank)

        worker_id = lane.target_worker._actor_id.hex()
        if worker_id in target_workers:
            raise RuntimeError(f"[{group_name}] Duplicate target worker {worker_id}")
        target_workers.add(worker_id)


def is_lane_participant(my_rank: int, lane: LaneRoute) -> bool:
    return my_rank in lane.source_ranks


def is_lane_leader(my_rank: int, lane: LaneRoute) -> bool:
    return my_rank == lane.leader_rank
