import torch
from megatron.core import mpu


def _megatron_calc_global_rank(
    tp_rank: int = 0,
    dp_rank: int = 0,
    pp_rank: int = 0,
    cp_rank: int = 0,
    ep_rank: int = 0,
):
    """Calculate global rank with support for CP/EP parallelism"""

    # Get parallel sizes for each dimension
    tp_size = mpu.get_tensor_model_parallel_world_size()
    dp_size = mpu.get_data_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    # ep_size = mpu.get_expert_model_parallel_world_size()

    # Verify total GPU count matches (must be consistent with parallel_state.py)
    total_size = tp_size * dp_size * pp_size * cp_size
    assert (
        total_size == torch.distributed.get_world_size()
    ), f"{tp_size}x{dp_size}x{pp_size}x{cp_size} != {torch.distributed.get_world_size()}"

    # Core calculation logic (corresponds to RankGenerator order parameter)
    # Assumes default order is "tp-cp-ep-dp-pp"
    return ((pp_rank * dp_size + dp_rank) * cp_size + cp_rank) * tp_size + tp_rank
