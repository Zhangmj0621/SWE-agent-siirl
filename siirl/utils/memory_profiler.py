# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
"""GPU Memory Profiler for debugging memory usage during training.

Environment variables:
    SIIRL_MEMORY_DEBUG=1     Enable detailed memory logging
    SIIRL_MEMORY_PROFILE=1   Export memory snapshots

Usage:
    from siirl.utils.memory_profiler import log_memory, memory_trace

    log_memory("Before forward", reset_peak=True)
    with memory_trace("compute_log_prob"):
        output = model(...)

    # Or analyze a snapshot file:
    python -m siirl.utils.memory_profiler snapshot.pickle
"""

import gc
import os
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import torch
from loguru import logger

__all__ = [
    "start_memory_recording",
    "stop_memory_recording_and_export",
    "get_memory_stats",
    "log_memory",
    "memory_trace",
    "profile_with_memory",
]

# ============================================================================
# Memory Recording (Detailed memory allocation tracking)
# ============================================================================


def start_memory_recording():
    """Start recording memory allocation history.

    This tracks the call stack for every memory allocation, useful for
    pinpointing memory usage sources.
    Note: Has performance overhead, only use during profiling.
    """
    if not torch.cuda.is_available():
        logger.debug("CUDA not available, skipping memory recording")
        return

    torch.cuda.memory._record_memory_history(
        max_entries=100000,
        context="all",  # Record both Python and C++ call stacks
    )
    logger.info("[Memory Profiler] Started memory recording with full stack traces")


def stop_memory_recording_and_export(output_path: str = "memory_snapshot.pickle"):
    """Stop recording and export memory snapshot.

    The exported file can be analyzed with PyTorch Memory Visualizer:
    https://pytorch.org/memory_viz

    Usage:
    1. Open https://pytorch.org/memory_viz
    2. Upload the generated .pickle file
    3. View allocation locations and call stacks for each tensor
    """
    if not torch.cuda.is_available():
        return

    try:
        snapshot = torch.cuda.memory._snapshot()
        if snapshot:
            import pickle

            with open(output_path, "wb") as f:
                pickle.dump(snapshot, f)
            logger.info(f"[Memory Profiler] Exported snapshot to {output_path}")
            logger.info("[Memory Profiler] Analyze at: https://pytorch.org/memory_viz")
        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception as e:
        logger.error(f"[Memory Profiler] Failed to export snapshot: {e}")


# ============================================================================
# PyTorch Profiler (Operation-level memory tracking)
# ============================================================================


@contextmanager
def profile_with_memory(output_dir: str = "./profiler_output", wait: int = 1, warmup: int = 1, active: int = 3):
    """Use PyTorch Profiler for memory profiling.

    Generated trace files can be opened in Chrome (chrome://tracing)
    or viewed in TensorBoard.

    Args:
        output_dir: Output directory for trace files
        wait: Number of steps to wait before profiling
        warmup: Number of warmup steps
        active: Number of steps to actually profile
    """
    if not torch.cuda.is_available():
        yield
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,  # Key: track memory
        with_stack=True,  # Key: record call stacks
    ) as prof:
        yield prof

    # Export Chrome trace
    chrome_trace_path = os.path.join(output_dir, f"trace_{timestamp}.json")
    prof.export_chrome_trace(chrome_trace_path)
    logger.info(f"[Memory Profiler] Exported Chrome trace to {chrome_trace_path}")

    # Print top memory consuming operations
    logger.info("[Memory Profiler] Top 10 memory consuming operations:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


# ============================================================================
# Simple Memory Statistics (Lightweight)
# ============================================================================


def get_memory_stats() -> dict:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }


def log_memory(tag: str = "", reset_peak: bool = False):
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return

    stats = get_memory_stats()
    logger.info(f"[Memory] {tag}")
    logger.info(f"  Allocated: {stats['allocated_gb']:.2f} GB | Reserved: {stats['reserved_gb']:.2f} GB")
    logger.info(f"  Peak Allocated: {stats['max_allocated_gb']:.2f} GB | Peak Reserved: {stats['max_reserved_gb']:.2f} GB")

    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
        logger.debug("  (Peak stats reset)")


@contextmanager
def memory_trace(tag: str):
    """Context manager to trace memory usage of a code block."""
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    start_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    yield

    torch.cuda.synchronize()
    end_allocated = torch.cuda.memory_allocated()
    peak_allocated = torch.cuda.max_memory_allocated()

    delta_alloc = (end_allocated - start_allocated) / (1024**3)
    peak_delta = (peak_allocated - start_allocated) / (1024**3)

    logger.info(f"[Memory Trace] {tag}")
    logger.info(f"  Start: {start_allocated / (1024**3):.2f}GB -> End: {end_allocated / (1024**3):.2f}GB (delta: {delta_alloc:+.2f}GB)")
    logger.info(f"  Peak: {peak_allocated / (1024**3):.2f}GB (peak delta: {peak_delta:+.2f}GB from start)")


def get_tensor_memory_breakdown() -> dict:
    """Get memory breakdown by tensor size (requires torch >= 2.0)."""
    if not torch.cuda.is_available():
        return {}

    try:
        # Get all tensors on GPU
        tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensors.append(
                        {
                            "size": obj.numel() * obj.element_size(),
                            "shape": tuple(obj.shape),
                            "dtype": str(obj.dtype),
                            "device": str(obj.device),
                        }
                    )
            except Exception:
                pass

        # Sort by size
        tensors.sort(key=lambda x: x["size"], reverse=True)

        # Get top 10
        total_size = sum(t["size"] for t in tensors)

        return {
            "total_tensor_gb": total_size / (1024**3),
            "num_tensors": len(tensors),
            "top_10": tensors[:10],
        }
    except Exception as e:
        logger.debug(f"Failed to get tensor breakdown: {e}")
        return {}


def log_tensor_breakdown(top_n: int = 10):
    """Log the largest tensors on GPU."""
    breakdown = get_tensor_memory_breakdown()
    if not breakdown:
        return

    logger.info("[Tensor Memory Breakdown]")
    logger.info(f"  Total tensors: {breakdown['num_tensors']} | Total size: {breakdown['total_tensor_gb']:.2f} GB")
    logger.info(f"  Top {top_n} tensors:")

    for i, t in enumerate(breakdown.get("top_10", [])[:top_n]):
        size_mb = t["size"] / (1024**2)
        logger.info(f"    {i+1}. {size_mb:.1f} MB - shape={t['shape']} dtype={t['dtype']}")


def print_memory_summary():
    """Print PyTorch's built-in memory summary."""
    if not torch.cuda.is_available():
        return

    logger.info("=" * 60)
    logger.info("[PyTorch CUDA Memory Summary]")
    logger.info("=" * 60)
    print(torch.cuda.memory_summary())


# ============================================================================
# Memory Snapshot Analysis Tool
# ============================================================================


def analyze_memory_snapshot(snapshot_path: str, top_n: int = 20):
    """Analyze memory snapshot file and print largest allocations.

    Args:
        snapshot_path: Path to memory snapshot file (.pickle)
        top_n: Number of top allocations to display
    """
    import pickle

    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)

    # Analyze segments
    total_allocated = 0
    total_reserved = 0
    allocations = []

    for seg in snapshot.get("segments", []):
        total_reserved += seg.get("total_size", 0)
        for block in seg.get("blocks", []):
            if block.get("state") == "active_allocated":
                size = block.get("size", 0)
                total_allocated += size

                # Get call stack
                frames = block.get("frames", [])
                if frames:
                    # Get recent frames
                    stack_info = []
                    for frame in frames[:5]:
                        filename = frame.get("filename", "?")
                        line = frame.get("line", 0)
                        name = frame.get("name", "?")
                        # Simplify path
                        if "site-packages" in filename:
                            filename = filename.split("site-packages/")[-1]
                        elif "siirl" in filename:
                            filename = filename.split("siirl/")[-1]
                        stack_info.append(f"{filename}:{line} ({name})")

                    allocations.append(
                        {
                            "size": size,
                            "stack": stack_info,
                        }
                    )

    # Sort by size
    allocations.sort(key=lambda x: x["size"], reverse=True)

    print("=" * 80)
    print(f"Memory Snapshot Analysis: {snapshot_path}")
    print("=" * 80)
    print(f"Total Allocated: {total_allocated / (1024**3):.2f} GB")
    print(f"Total Reserved:  {total_reserved / (1024**3):.2f} GB")
    print(f"Num Allocations: {len(allocations)}")
    print()
    print(f"Top {top_n} Largest Allocations:")
    print("-" * 80)

    for i, alloc in enumerate(allocations[:top_n]):
        size_mb = alloc["size"] / (1024**2)
        print(f"\n{i+1}. Size: {size_mb:.1f} MB")
        for frame in alloc["stack"]:
            print(f"     {frame}")

    # Aggregate by file
    print("\n" + "=" * 80)
    print("Memory by Source File:")
    print("-" * 80)

    by_file = defaultdict(int)
    for alloc in allocations:
        if alloc["stack"]:
            # Find first siirl-related file
            for frame in alloc["stack"]:
                if "siirl" in frame or "megatron" in frame:
                    file_name = frame.split(":")[0]
                    by_file[file_name] += alloc["size"]
                    break
            else:
                # Not found, use first frame
                file_name = alloc["stack"][0].split(":")[0]
                by_file[file_name] += alloc["size"]

    sorted_files = sorted(by_file.items(), key=lambda x: x[1], reverse=True)
    for file_name, size in sorted_files[:15]:
        print(f"  {size / (1024**2):8.1f} MB  {file_name}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        analyze_memory_snapshot(sys.argv[1])
    else:
        print("Usage: python memory_profiler.py <snapshot.pickle>")
