import json
import os
from typing import Optional

import torch
import torch.distributed as dist
from tensordict import TensorDict

from siirl.engine.actor.megatron_actor import ActorWorker
from siirl.params.model_args import ActorArguments, ActorRefArguments, MegatronArguments, ModelArguments, OptimizerArguments

# ============================================================================
# Constants and Configuration
# ============================================================================

GLOBAL_BATCH_SIZE = 8
SEED = 1234
SEQ_LEN = 128
RESPONSE_LEN = 32
TOLERANCE = 1e-3
BASELINE_DIR = "test_baselines"


# ============================================================================
# Utility Functions
# ============================================================================


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def print_rank0(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def cleanup_parallel_groups():
    """Clean up both Megatron parallel groups and torch.distributed

    CRITICAL: Must destroy torch.distributed to clear NCCL communicator state.
    Without this, residual NCCL state from previous tests (especially CP groups)
    will cause deadlocks when trying to create new groups with different sizes.
    """
    try:
        from megatron.core import parallel_state as mpu

        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    except Exception as e:
        print(f"Warning: Failed to destroy Megatron groups: {e}")


# ============================================================================
# Configuration Factory
# ============================================================================


class ConfigFactory:
    """Factory for creating test configurations with consistent global batch size"""

    def __init__(self, model_path: str, global_batch_size: int = GLOBAL_BATCH_SIZE):
        self.model_path = model_path
        self.global_batch_size = global_batch_size

    def create(
        self,
        tp: int = 1,
        pp: int = 1,
        cp: int = 1,
        dp: int = 1,
        sp: bool = False,
        seed: int = SEED,
    ) -> ActorRefArguments:
        """Create configuration ensuring global batch size consistency"""

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        required_world_size = tp * pp * cp * dp

        if world_size != required_world_size:
            raise ValueError(
                f"WORLD_SIZE mismatch: config requires {required_world_size} GPUs "
                f"(TP={tp} × PP={pp} × CP={cp} × DP={dp}), "
                f"but environment has WORLD_SIZE={world_size}"
            )

        self._setup_distributed_env()

        ppo_mini_batch_size = self.global_batch_size * dp

        return ActorRefArguments(
            hybrid_engine=False,
            model=ModelArguments(
                path=self.model_path,
                trust_remote_code=False,
            ),
            actor=ActorArguments(
                ppo_mini_batch_size=ppo_mini_batch_size,
                ppo_micro_batch_size_per_gpu=2,
                ppo_epochs=1,
                n=1,
                temperature=1.0,
                entropy_coeff=0.01,
                clip_ratio=0.2,
                use_kl_loss=False,
                load_weight=False,
                megatron=MegatronArguments(
                    tensor_model_parallel_size=tp,
                    pipeline_model_parallel_size=pp,
                    context_parallel_size=cp,
                    virtual_pipeline_model_parallel_size=None,
                    sequence_parallel=sp,
                    use_distributed_optimizer=False,
                    param_dtype="bfloat16",
                    seed=seed,
                    param_offload=False,
                    grad_offload=False,
                    optimizer_offload=False,
                    use_mbridge=False,
                ),
                optim=OptimizerArguments(
                    lr=1e-5,
                    lr_warmup_steps=10,
                    total_training_steps=100,
                    clip_grad=1.0,
                    weight_decay=0.01,
                    override_optimizer_config={"use_distributed_optimizer": False},
                ),
            ),
        )

    @staticmethod
    def _setup_distributed_env():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))

        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")


# ============================================================================
# Data Generator
# ============================================================================


class DataGenerator:
    """Generate deterministic training data for reproducibility"""

    @staticmethod
    def create(
        batch_size: int,
        seq_len: int = SEQ_LEN,
        response_len: int = RESPONSE_LEN,
        seed: int = SEED,
    ) -> TensorDict:
        set_seed(seed)

        input_ids = torch.randint(100, 32000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        responses = input_ids[:, -response_len:]
        response_mask = torch.ones((batch_size, response_len), dtype=torch.float32)
        old_log_probs = torch.randn((batch_size, response_len), dtype=torch.float32) * 0.1
        advantages = torch.randn((batch_size, response_len), dtype=torch.float32)

        return TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
                "old_log_probs": old_log_probs,
                "advantages": advantages,
            },
            batch_size=(batch_size,),
        )


# ============================================================================
# Metrics Manager
# ============================================================================


class MetricsManager:
    """Handle saving, loading, and validating metrics"""

    def __init__(self, baseline_dir: str = BASELINE_DIR):
        self.baseline_dir = baseline_dir

    def save(self, metrics: dict, config_name: str):
        if not is_rank0():
            return

        os.makedirs(self.baseline_dir, exist_ok=True)

        metrics_serializable = self._serialize_metrics(metrics)
        metrics_path = os.path.join(self.baseline_dir, f"{config_name}_baseline.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)

    def load(self, config_name: str):
        metrics_path = os.path.join(self.baseline_dir, f"{config_name}_baseline.json")

        if not os.path.exists(metrics_path):
            return None

        with open(metrics_path) as f:
            metrics = json.load(f)

        return metrics

    def validate(
        self,
        metrics: dict,
        baseline_metrics: dict,
        tolerance: float = TOLERANCE,
        validate_keys: list[str] | None = None,
    ) -> bool:
        if baseline_metrics is None:
            return True

        all_passed = True
        print_rank0("Validation:")

        for key in sorted(metrics.keys()):
            if key not in baseline_metrics:
                continue

            # Skip validation if validate_keys is specified and key is not in the list
            if validate_keys is not None and key not in validate_keys:
                print_rank0(f"  ⊘ {key}: (skipped)")
                continue

            current_val = self._extract_value(metrics[key])
            baseline_val = self._extract_value(baseline_metrics[key])

            diff = abs(current_val - baseline_val)
            rel_diff = diff / (abs(baseline_val) + 1e-10)

            passed = diff < tolerance or rel_diff < tolerance
            status = "✓" if passed else "✗"

            print_rank0(f"  {status} {key}: current={current_val:.8f}, baseline={baseline_val:.8f}, diff={diff:.8e}")

            if not passed:
                all_passed = False

        return all_passed

    @staticmethod
    def _serialize_metrics(metrics: dict) -> dict:
        return {key: value[0] if isinstance(value, list) else value for key, value in metrics.items()}

    @staticmethod
    def _extract_value(value):
        return value[0] if isinstance(value, list) else value


# ============================================================================
# Training Runner
# ============================================================================


class TrainingRunner:
    """Execute training runs and collect metrics"""

    def __init__(self, config_factory: ConfigFactory, data_generator: DataGenerator):
        self.config_factory = config_factory
        self.data_generator = data_generator

    def run(
        self,
        tp: int = 1,
        pp: int = 1,
        cp: int = 1,
        dp: int = 1,
        sp: bool = False,
        seed: int = SEED,
        input_data: TensorDict | None = None,
    ) -> dict | None:
        """Run training and return metrics"""

        cleanup_parallel_groups()

        config = self.config_factory.create(tp=tp, pp=pp, cp=cp, dp=dp, sp=sp, seed=seed)

        try:
            set_seed(seed)
            actor_worker = ActorWorker(config)
            actor_worker.init_model()

            if input_data is None:
                batch_size = config.actor.ppo_mini_batch_size
                input_data = self.data_generator.create(batch_size=batch_size, seed=seed)

            set_seed(seed)
            result = actor_worker.update_actor(input_data)

            if "metrics" not in result:
                return None

            metrics = self._broadcast_metrics(result["metrics"], pp)
            # Do NOT cleanup here - let caller handle cleanup

            return metrics

        except Exception as e:
            print_rank0(f"Training failed: {e}")
            return None

    @staticmethod
    def _broadcast_metrics(metrics: dict, pp: int) -> dict:
        """Broadcast metrics from last pipeline stage to all ranks"""
        if not dist.is_initialized() or pp <= 1:
            return metrics

        try:
            from megatron.core import parallel_state as mpu

            if mpu.is_initialized() and mpu.get_pipeline_model_parallel_world_size() > 1:
                last_pp_rank = mpu.get_pipeline_model_parallel_last_rank()

                if dist.get_rank() == last_pp_rank:
                    metrics_list = [metrics]
                else:
                    metrics_list = [None]

                dist.broadcast_object_list(metrics_list, src=last_pp_rank)
                return metrics_list[0]
        except Exception:
            pass

        return metrics


# ============================================================================
# Test Orchestrator
# ============================================================================


class TestOrchestrator:
    """Orchestrate the testing workflow"""

    def __init__(
        self,
        config_factory: ConfigFactory,
        training_runner: TrainingRunner,
        metrics_manager: MetricsManager,
    ):
        self.config_factory = config_factory
        self.training_runner = training_runner
        self.metrics_manager = metrics_manager

    def run_baseline(self, config_name: str = "baseline") -> bool:
        """Run baseline single-GPU experiment and save results"""

        print_rank0("Running baseline initialization...")

        # Run training (input_data will be generated with seed inside)
        metrics = self.training_runner.run(tp=1, pp=1, cp=1, dp=1, sp=False, seed=SEED)

        if metrics is None:
            print_rank0("ERROR: Baseline training failed")
            return False

        self._print_metrics(metrics)

        # Save only the metrics (input_data will be regenerated with same seed during validation)
        self.metrics_manager.save(metrics, config_name)

        print_rank0("Baseline saved successfully")
        return True

    def validate_config(
        self,
        config_name: str,
        tp: int,
        pp: int,
        cp: int,
        dp: int,
        sp: bool,
        baseline_name: str = "baseline",
        validate_keys: list[str] | None = None,
    ) -> bool:
        """Validate a parallel configuration against baseline"""

        baseline_metrics = self.metrics_manager.load(baseline_name)

        if baseline_metrics is None:
            print_rank0("ERROR: No baseline found. Run with --init first.")
            return False

        # Regenerate input_data with same seed (not loaded from disk)
        # This ensures all ranks have identical data without file I/O issues
        metrics = self.training_runner.run(tp=tp, pp=pp, cp=cp, dp=dp, sp=sp, seed=SEED)

        if metrics is None:
            print_rank0("ERROR: Training failed")
            return False

        self._print_metrics(metrics)

        validation_passed = self.metrics_manager.validate(metrics, baseline_metrics, validate_keys=validate_keys)

        return validation_passed

    @staticmethod
    def _print_metrics(metrics: dict):
        print_rank0("Metrics:")
        for key, value in sorted(metrics.items()):
            val = value[0] if isinstance(value, list) else value
            print_rank0(f"  {key}: {val:.8f}")


# ============================================================================
# Test Suite Definitions
# ============================================================================


class TestSuite:
    """Define test configurations organized by parallelism strategy"""

    @staticmethod
    def get_configs() -> list[tuple[str, int, int, int, int, bool, int]]:
        """Return test configs: (name, tp, pp, cp, dp, sp, required_gpus)"""

        return [
            # Baseline
            ("baseline", 1, 1, 1, 1, False, 1),
            # Single parallelism strategy
            ("tp2", 2, 1, 1, 1, False, 2),
            ("tp4", 4, 1, 1, 1, False, 4),
            ("tp8", 8, 1, 1, 1, False, 8),
            ("pp2", 1, 2, 1, 1, False, 2),
            ("pp4", 1, 4, 1, 1, False, 4),
            ("pp8", 1, 8, 1, 1, False, 8),
            ("cp2", 1, 1, 2, 1, True, 2),
            ("cp4", 1, 1, 4, 1, True, 4),
            ("dp2", 1, 1, 1, 2, False, 2),
            ("dp4", 1, 1, 1, 4, False, 4),
            ("dp8", 1, 1, 1, 8, False, 8),
            # Mixed parallelism
            ("tp2_pp2", 2, 2, 1, 1, False, 4),
            ("tp2_pp4", 2, 4, 1, 1, False, 8),
            ("tp4_pp2", 4, 2, 1, 1, False, 8),
            ("tp2_cp2", 2, 1, 2, 1, True, 4),
            ("tp2_dp2", 2, 1, 1, 2, False, 4),
            ("tp2_dp4", 2, 1, 1, 4, False, 8),
            ("tp4_dp2", 4, 1, 1, 2, False, 8),
            ("pp2_dp2", 1, 2, 1, 2, False, 4),
            ("pp2_dp4", 1, 2, 1, 4, False, 8),
            ("pp4_dp2", 1, 4, 1, 2, False, 8),
            ("cp2_dp2", 1, 1, 2, 2, True, 4),
            ("cp2_dp4", 1, 1, 2, 4, True, 8),
            # Complex mixed parallelism
            ("tp2_pp2_dp2", 2, 2, 1, 2, False, 8),
            ("tp2_pp2_cp2", 2, 2, 2, 1, True, 8),
            # Sequence parallelism (requires TP > 1)
            ("tp2_sp", 2, 1, 1, 1, True, 2),
            ("tp4_sp", 4, 1, 1, 1, True, 4),
            ("tp2_dp2_sp", 2, 1, 1, 2, True, 4),
            ("tp2_pp2_sp", 2, 2, 1, 1, True, 4),
        ]

    @staticmethod
    def filter_by_world_size(world_size: int) -> list[tuple]:
        """Filter configs matching current world size"""
        return [config for config in TestSuite.get_configs() if config[6] == world_size]


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Actor training accuracy validation")
    parser.add_argument("--init", action="store_true", help="Initialize baseline")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Test specific config (required for validation)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the HuggingFace model (for tokenizer and config)",
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # num_gpus = torch.cuda.device_count()

    # Only validate pgloss (gradnorm cannot be aligned)
    validate_keys = ["actor/pg_loss"]

    config_factory = ConfigFactory(args.model_path, GLOBAL_BATCH_SIZE)
    data_generator = DataGenerator()
    training_runner = TrainingRunner(config_factory, data_generator)
    metrics_manager = MetricsManager()
    orchestrator = TestOrchestrator(config_factory, training_runner, metrics_manager)

    exit_code = 0

    try:
        if args.init:
            if world_size != 1:
                print_rank0("ERROR: Baseline initialization requires single GPU")
                print_rank0("Run with: python test_actor_training.py --init")
                exit_code = 1
                return

            success = orchestrator.run_baseline()

            if success:
                print_rank0("✓ BASELINE INITIALIZATION COMPLETED")
            else:
                print_rank0("✗ BASELINE INITIALIZATION FAILED")
                exit_code = 1

            return

        # Validation mode - require config argument
        if not args.config:
            print_rank0("ERROR: --config is required for validation mode")
            print_rank0("Usage: python test_actor_training.py --config <config_name>")
            print_rank0("Available configs:")
            for (
                config_name,
                tp,
                pp,
                cp,
                dp,
                sp,
                required_gpus,
            ) in TestSuite.get_configs():
                print_rank0(f"  {config_name:20s} (TP={tp}, PP={pp}, CP={cp}, DP={dp}, SP={sp}, GPUs={required_gpus})")
            exit_code = 1
            return

        # Find the specified config
        all_configs = TestSuite.get_configs()
        configs = [c for c in all_configs if c[0] == args.config]

        if not configs:
            print_rank0(f"ERROR: Config '{args.config}' not found")
            print_rank0(f"Available: {', '.join([c[0] for c in all_configs])}")
            exit_code = 1
            return

        config_name, tp, pp, cp, dp, sp, required_gpus = configs[0]

        # Check world size matches
        if world_size != required_gpus:
            print_rank0(f"ERROR: Config '{config_name}' requires {required_gpus} GPUs, but world_size={world_size}")
            exit_code = 1
            return

        # Skip baseline in validation mode
        if config_name == "baseline":
            print_rank0("Baseline is initialized with --init, skipping validation")
            return

        # Run single test
        print_rank0(f"Testing: {config_name} (TP={tp}, PP={pp}, CP={cp}, DP={dp}, SP={sp})")
        success = orchestrator.validate_config(config_name, tp, pp, cp, dp, sp, validate_keys=validate_keys)

        cleanup_parallel_groups()

        if success:
            print_rank0(f"✓ {config_name} PASSED")
        else:
            print_rank0(f"✗ {config_name} FAILED")
            exit_code = 1  # All ranks set the same exit code

    finally:
        # Always destroy process group before exit
        if dist.is_initialized():
            dist.destroy_process_group()

    # Exit with appropriate code
    if exit_code != 0:
        exit(exit_code)


if __name__ == "__main__":
    main()
