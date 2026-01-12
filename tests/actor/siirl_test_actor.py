"""
Test script for ActorWorker training
"""

import argparse
import math
import os
from abc import abstractmethod
from contextlib import contextmanager

import torch
from loguru import logger
from tensordict import MemoryMappedTensor, TensorDict

from siirl.engine.actor.megatron_actor import ActorWorker, CriticWorker, ReferenceWorker
from siirl.params.model_args import (
    ActorArguments,
    ActorRefArguments,
    CriticArguments,
    MegatronArguments,
    ModelArguments,
    OptimizerArguments,
    RefArguments,
)


@contextmanager
def temporary_env(env_dict: dict):
    """临时设置环境变量，退出时自动清理或恢复原值"""
    original_value = {}
    for key, value in env_dict.items():
        original_value[key] = os.environ.get(key, None)
        os.environ[key] = str(value)
    logger.info(f"Now Enter {env_dict}")
    try:
        yield
    finally:
        logger.info(f"Now Exit {env_dict}")
    for key, value in original_value.items():
        if value is None:
            del os.environ[key]
        else:
            os.environ[key] = value


def create_grpo_config(tp, pp, model_path):
    """Create a test configuration for ActorWorker"""
    # Create configuration
    config = ActorRefArguments(
        hybrid_engine=False,
        model=ModelArguments(
            path=model_path,
            trust_remote_code=False,
        ),
        actor=ActorArguments(
            ppo_mini_batch_size=32,
            ppo_micro_batch_size_per_gpu=8,
            ppo_epochs=1,
            n=8,
            temperature=1.0,
            entropy_coeff=0.01,
            # grad_clip
            clip_ratio=0.2,
            kl_loss_coef=0.01,
            kl_loss_type="low_var_kl",
            use_kl_loss=True,
            load_weight=True,
            megatron=MegatronArguments(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                sequence_parallel=False,
                use_distributed_optimizer=False,
                param_dtype="bfloat16",
                seed=1,
                param_offload=True,
                grad_offload=True,
                optimizer_offload=True,
                use_mbridge=False,
            ),
            optim=OptimizerArguments(
                lr=1e-6,
                lr_warmup_steps=10,
                total_training_steps=100,
                clip_grad=1.0,
                weight_decay=0.01,
                override_optimizer_config={"use_distributed_optimizer": False},
            ),
        ),
        ref=RefArguments(
            log_prob_micro_batch_size_per_gpu=8,
            param_offload=False,
        ),
    )

    return config


def create_ppo_config(tp, pp, model_path):
    """Create a test configuration for ActorWorker"""
    # Create configuration
    config = ActorRefArguments(
        hybrid_engine=False,
        model=ModelArguments(
            path=model_path,
            trust_remote_code=False,
        ),
        actor=ActorArguments(
            ppo_mini_batch_size=32,
            ppo_micro_batch_size_per_gpu=8,
            ppo_epochs=1,
            n=1,
            temperature=1.0,
            entropy_coeff=0.01,
            clip_ratio=0.2,
            kl_loss_coef=0.01,
            kl_loss_type="low_var_kl",
            use_kl_loss=True,
            load_weight=True,
            megatron=MegatronArguments(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                sequence_parallel=False,
                use_distributed_optimizer=False,
                param_dtype="bfloat16",
                seed=1,
                param_offload=True,
                grad_offload=True,
                optimizer_offload=True,
                use_mbridge=True,
            ),
            optim=OptimizerArguments(
                lr=1e-6,
                lr_warmup_steps=10,
                total_training_steps=100,
                clip_grad=1.0,
                weight_decay=0.01,
                override_optimizer_config={"use_distributed_optimizer": False},
            ),
        ),
        ref=RefArguments(
            log_prob_micro_batch_size_per_gpu=8,
            param_offload=False,
        ),
    )
    critci_config = CriticArguments(
        ppo_mini_batch_size=32,
        ppo_micro_batch_size_per_gpu=32,
        ppo_epochs=1,
        ppo_max_token_len_per_gpu=98304,
        load_weight=True,
        model=ModelArguments(
            path=model_path,
            trust_remote_code=False,
        ),
        megatron=MegatronArguments(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
            use_distributed_optimizer=True,
            param_dtype="bfloat16",
            seed=1,
            param_offload=True,
            grad_offload=True,
            optimizer_offload=True,
            use_mbridge=False,
        ),
        optim=OptimizerArguments(
            lr=1e-5,
            total_training_steps=100,
        ),
    )
    return config, critci_config


class TensorDictLoader:
    def __init__(self, dump_data_path):
        from megatron.core import parallel_state as mpu

        self.rank = torch.distributed.get_rank()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.dp_rank = mpu.get_data_parallel_rank()
        self.base_path = dump_data_path

    def _get_pt_file_path(self, save_count=0, prefix="", type="input"):
        filename_suffix = (
            f"save_count_{save_count}_{prefix}_type_{type}_tensor_dict_"
            f"rank_{self.rank}_tp_{self.tp_rank}_pp_{self.pp_rank}_dp_{self.dp_rank}"
        )
        # logger.info(f"loading pt file:{filename_suffix}")
        return os.path.join(self.base_path, filename_suffix)

    def load_tensordict(self, save_count=0, prefix="", type="input"):
        tensor_dict = TensorDict.load(self._get_pt_file_path(save_count, prefix, type))
        batch_size = tensor_dict["input_ids"].shape[0]
        seq_len = tensor_dict["input_ids"].shape[1]
        response_len = tensor_dict["responses"].shape[1]
        return tensor_dict, batch_size, seq_len, response_len


def compare_matrix(current, reference, keywords=None):
    if keywords is None:
        keywords = ["metrics"]
    if not isinstance(keywords, (list, tuple)):
        keywords = [keywords]

    for keyword in keywords:
        current_ = current[keyword]
        baseline_ = reference[keyword]
        check_precision(keyword, current_, baseline_)
        # print_metics(current_,"Training")
        # print_metics(baseline_,"Baseline")


def print_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, print only on rank 0."""
    if rank is not None:
        if rank == 0:
            logger.info(message, flush=True)
    elif torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(message, flush=True)
    else:
        logger.info(message, flush=True)


def check_precision(name, current, reference, rtol=1e-5, atol=1e-8) -> bool:
    """
    比较单个值（Tensor, int, float）的精度。

    Args:
        name (str): 变量名，用于日志标识
        current: 当前值
        reference: 参考值 (通常来自pt文件)
        rtol: 相对误差
        atol: 绝对误差

    Returns:
        bool: 精度是否匹配 (True/False)
    """
    # try:
    # 获取 rank 仅用于日志打印
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    if isinstance(reference, MemoryMappedTensor):
        reference = torch.Tensor(reference)
    # 1. 类型检查
    # 如果类型根本不一致（例如一个是Tensor一个是int），直接返回 False
    if type(current) is not type(reference):
        logger.warning(f"❌ Rank {rank} [{name}] Type mismatch: current {type(current)} vs ref {type(reference)}")
        return False

    # 2. Tensor 比较
    if isinstance(current, torch.Tensor):
        current = current.cpu()
        reference = reference.cpu()

        # 形状检查
        if current.shape != reference.shape:
            logger.warning(f"❌ Rank {rank} [{name}] Shape mismatch: current {current.shape} vs ref {reference.shape}")
            return False

        # 精度检查
        if not torch.allclose(current, reference, rtol=rtol, atol=atol):
            diff = (current - reference).abs().max()
            logger.warning(f"❌ Rank {rank} [{name}] Value mismatch: max diff {diff:.6f}")
            return False

        logger.info(f"✅ Rank {rank} [{name}] Tensor match")
        return True

    # 3. 标量比较 (int/float)
    elif isinstance(current, (int, float)):
        if not math.isclose(current, reference, rel_tol=rtol, abs_tol=atol):
            diff = abs(current - reference)
            logger.warning(
                f"❌ Rank {rank} [{name}] Scalar mismatch: cur {current} vs ref {reference}, diff {diff:.6f}"
            )
            return False

        logger.info(f"✅ Rank {rank} [{name}] Scalar match")
        return True

    # 4. 其他类型直接比较 (str, bool, None等)
    else:
        if current != reference:
            logger.warning(f"❌ Rank {rank} [{name}] Value mismatch: cur {current} vs ref {reference}")
            return False

        logger.info(f"✅ Rank {rank} [{name}] Obj match")
        return True

    # except Exception as e:
    #     logger.error(f"Error checking precision for {name}: {e}")
    #     return False


def get_data_and_baseline(loader, current_step, prefix):
    data, batch_size, seq_len, response_len = loader.load_tensordict(current_step, prefix)
    baseline, _, _, _ = loader.load_tensordict(current_step, prefix, type="output")
    print_rank_0(f"   - Batch size: {batch_size}")
    print_rank_0(f"   - Sequence length: {seq_len}")
    print_rank_0(f"   - Response length: {response_len}")
    print_rank_0(f"   - Data keys: {list(baseline.keys())}")
    return data, baseline


class BaseTrainer:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def init_worker(self):
        pass

    @abstractmethod
    def run_step(self, current_step):
        pass


class GRPOTrainer(BaseTrainer):
    def __init__(self, config, data_path):
        super().__init__(config)
        self.actor_worker = ActorWorker(self.config)
        self.ref_worker = ReferenceWorker(self.config)
        self.loader = TensorDictLoader(data_path)

    def init_worker(self):
        try:
            self.actor_worker.init_model()
            self.ref_worker.init_model()
            logger.info("   ✓ Model initialized successfully")
        except Exception as e:
            logger.info(f"   ✗ Failed to initialize model: {e}")
            import traceback

            traceback.print_exc()
            raise e

    def run_step(self, current_step):

        # with temporary_env({"siirl_status":"compute_ref_log_prob","step":str(current_step)}):
        data, baseline = get_data_and_baseline(self.loader, current_step, "reference_log_prob")
        ref_log_prob = self.ref_worker.compute_ref_log_prob(data)
        compare_matrix(ref_log_prob, baseline, "ref_log_prob")

        # with temporary_env({"siirl_status":"compute_log_prob","step":str(current_step)}):
        data, baseline = get_data_and_baseline(self.loader, current_step, "actor_old_log_prob")
        old_log_prob = self.actor_worker.compute_log_prob(data)
        compare_matrix(old_log_prob, baseline, "old_log_probs")

        # with temporary_env({"siirl_status":"update_policy","step":str(current_step)}):
        data, baseline = get_data_and_baseline(self.loader, current_step, "actor_train")
        result = self.actor_worker.update_actor(data)
        compare_matrix(result["metrics"], baseline["metrics"], ["actor/pg_loss", "actor/ppo_kl"])


class PPOTrainer(BaseTrainer):
    def __init__(self, config, critic_config, data_path):
        super().__init__(config)
        self.actor_worker = ActorWorker(self.config)
        self.critic = CriticWorker(critic_config)
        self.ref_worker = ReferenceWorker(self.config)
        self.loader = TensorDictLoader(data_path)

    def init_worker(self):
        self.critic.init_model()
        self.ref_worker.init_model()
        self.actor_worker.init_model()

    def run_step(self, current_step):

        data, baseline = get_data_and_baseline(self.loader, current_step, "critic_compute_value")
        compute_values = self.critic.compute_values(data)
        compare_matrix(compute_values, baseline, "values")

        data, baseline = get_data_and_baseline(self.loader, current_step, "reference_log_prob")
        ref_log_prob = self.ref_worker.compute_ref_log_prob(data)
        compare_matrix(ref_log_prob, baseline, "ref_log_prob")

        data, baseline = get_data_and_baseline(self.loader, current_step, "actor_old_log_prob")
        old_log_prob = self.actor_worker.compute_log_prob(data)
        compare_matrix(old_log_prob, baseline, "old_log_probs")

        data, baseline = get_data_and_baseline(self.loader, current_step, "actor_train")
        result = self.actor_worker.update_actor(data)
        compare_matrix(result["metrics"], baseline["metrics"], ["actor/pg_loss", "actor/ppo_kl"])

        data, baseline = get_data_and_baseline(self.loader, current_step, "critic_train")
        result = self.critic.update_critic(data)
        compare_matrix(
            result["metrics"],
            baseline["metrics"],
            ["critic/vf_loss", "critic/vpred_mean"],
        )


def main():
    """Main test function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--test-step", type=int, default=1, help="Step num")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/inspire/ssd/project/qianghuaxuexi/public/debug_models/Qwen3-8B/",
        help="Path to Model",
    )
    parser.add_argument(
        "--tensordict-data-path",
        type=str,
        default="/inspire/ssd/project/qianghuaxuexi/public/debug_models/Qwen3-8B/",
        help="Path to Model",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "grpo"],
        default="grpo",
        help="Algorithm selection: 'ppo' or 'grpo'",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Actor Training Test Script")
    print("=" * 80)
    # Initialize ActorWorker
    print("\n2. Initializing ActorWorker...")
    trainer = None
    if args.algo == "grpo":
        config = create_grpo_config(tp=args.tp, pp=args.pp, model_path=args.model_path)
        trainer = GRPOTrainer(config, args.tensordict_data_path)
    elif args.algo == "ppo":
        config, critic_config = create_ppo_config(tp=args.tp, pp=args.pp, model_path=args.model_path)
        trainer = PPOTrainer(config, critic_config, args.tensordict_data_path)
    else:
        raise Exception()

    # Initialize model
    print_rank_0("\n3. Initializing model...")
    trainer.init_worker()

    total_step = args.test_step
    for current_step in range(total_step):
        # Run training step
        print_rank_0("\n4. Running training step...")
        try:
            trainer.run_step(current_step)
            print_rank_0("   ✓ Training step completed successfully")
            # Print metrics if available
        except Exception as e:
            print_rank_0(f"   ✗ Training step failed: {e}")
            import traceback

            traceback.print_exc()
            return

    print_rank_0("=" * 80)
    print_rank_0("Test completed successfully!")
    print_rank_0("=" * 80)


if __name__ == "__main__":
    main()
