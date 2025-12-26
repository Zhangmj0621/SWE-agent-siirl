from siirl.engine.actor.megatron_actor import ActorWorker,ReferenceWorker,CriticWorker
from siirl.worker.rollout.rollout_manager import RolloutManager
from siirl.params.model_args import (
    ActorRefArguments,
    ModelArguments,
    ActorArguments,
    MegatronArguments,
    OptimizerArguments,
    RefArguments,
    RolloutArguments,
)
from siirl.worker.ray_utils import allocate_resources
from siirl.params import SiiRLArguments,TrainingArguments
from siirl.engine.param_sync.update_weight import ParamSyncDistributed
from typing import  Optional, List
from siirl.engine.actor.utils import get_master_info
from siirl.utils.enums import DistributedEnv
import os
import ray
import argparse
from test_utils.test_trainer_group import TestTrainerGroup
import time
from siirl.engine.rollout.sglang_engine import SglangEngine
from siirl.utils.distributed_utils import init_gloo_group
import time
def create_grpo_config(tp, pp,rollout_tp, model_path):
    """Create a test configuration for ActorWorker"""
    # Create configuration
    config = ActorRefArguments(
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
            load_weight=False,
            megatron=MegatronArguments(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                sequence_parallel=False,
                use_distributed_optimizer=False,
                param_dtype="bfloat16",
                seed=1,
                param_offload=False,
                grad_offload=False,
                optimizer_offload=False,
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
        )
    )
    rollout_cfg = RolloutArguments(
        tensor_model_parallel_size=rollout_tp,
        temperature=0,
        top_k=-1,
        top_p=1,
        gpu_memory_utilization=0.4
    )
    trainer_args = TrainingArguments(n_gpus_per_node=8,nnodes=1,actor_gpus=tp*pp,rollout_gpus=0,colocate=False)
    return SiiRLArguments(actor_ref=config,rollout=rollout_cfg,trainer=trainer_args)

class TestTrainer:
    """
    Single training unit managing actor, reference, and optionally critic models.
    Each Trainer handles data fetching and training execution for one GPU.
    """

    def __init__(
        self,
        config,
        rank: int,
        local_rank: int,
        world_size: int,
        use_critic: bool = False,
        data_coordinator=None,
        coordinator=None,
        rollout_manager = None,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.data_coordinator = data_coordinator

        self.rollout_manager = rollout_manager

        # Initialize models (will be created in init_models method)
        self.actor_worker = None
        self.dp_rank = None
        self.dp_world_size = None

        # Log trainer initialization info
        from loguru import logger
        node_ip = ray.util.get_node_ip_address()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        ray_gpu_ids = ray.get_gpu_ids()
        logger.info(f"[Trainer.__init__] rank={rank}, local_rank={local_rank}, world_size={world_size}")
        logger.info(f"  node_ip={node_ip}, CUDA_VISIBLE_DEVICES={cuda_visible}, ray_gpu_ids={ray_gpu_ids}")

    def init_models(self):
        self.actor_worker = ActorWorker(config=self.config.actor_ref)
        self.actor_worker.init_model()

    def update_actor(self):
        self.actor_worker.actor_module
    
    def get_actor_model_param(self, param_num=3):
        self.actor_worker.bridge

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager

    def setup_param_sync(self):
        assert self.actor_worker is not None,"must init models first"
        self.param_sync = ParamSyncDistributed(config=self.config, model=self.actor_worker.actor_module, bridge=self.actor_worker.bridge)
        init_gloo_group()
    # @timer
    def update_rollout_weight(self):
        assert self.param_sync is not None, "must setup param sync first"
        if isinstance(self.param_sync,ParamSyncDistributed):
            # TODO support elastic rollout connection
            rollout_workers = ray.get(self.rollout_manager.get_workers.remote())
            if any(not self.param_sync.has_connected_to_actor(x) for x in rollout_workers):   
                self.param_sync.setup_param_sync_group(rollout_workers)
        self.param_sync.update_weights()


def create_config(args):
    return create_grpo_config(tp=args.tp, pp=args.pp,rollout_tp=args.rollout_tp,model_path=args.model_path)

@ray.remote
class TestRolloutManager():
    def __init__(self,config) -> None:
        self.config = config
        self.rollout_workers = self.start_sglang_worker()
    def start_sglang_worker(self):
        config = self.config
        RemoteSglangEngine = ray.remote(num_gpus=2)(SglangEngine)

        env_vars = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1"

        }
        extra_server_args = {
            "disable_radix_cache" : True
        }
        engine1 = RemoteSglangEngine.options(runtime_env={"env_vars": env_vars}).remote(
            rank=0,
            config=config,
            dist_init_addr="127.0.0.1:20001",
            ip="127.0.0.1",
            port=30001,
            nccl_port=40001,
            base_gpu_id=4,
            node_rank=0,
            nnodes=1,
            extra_server_args = extra_server_args,
        )

        engine2 = RemoteSglangEngine.options(runtime_env={"env_vars": env_vars}).remote(
            rank=1,
            config=config,
            dist_init_addr="127.0.0.1:20002",
            ip="127.0.0.1",
            port=30002,
            nccl_port=40002,
            base_gpu_id=6,
            node_rank=0,
            nnodes=1,
            extra_server_args = extra_server_args,
        )
        return engine1, engine2

    def get_workers(self):
        return self.rollout_workers

@ray.remote(num_cpus=5)
class MainRunner:
    """
    A Ray actor responsible for orchestrating the entire RL training workflow.

    This actor handles loading configurations, scheduling task graphs, initializing
    process groups, and launching the distributed Ray trainers. Isolating this
    orchestration logic in a dedicated actor ensures the main process remains clean
    and that the setup process is managed within the Ray cluster.
    """

    def run(self, config: SiiRLArguments) -> None:
        """
        Executes the main training workflow.

        Args:
            config: A SiiRLArguments object containing all parsed configurations.
        """
        from loguru import logger
        logger.info("Allocating GPU resources...")
        resources = allocate_resources(config)
        actor_resources = resources["actor"]
        rollout_resources = resources["rollout"]
        del rollout_resources
        config.trainer.rollout_gpus = 4
        # NOTE: Logging is automatically configured when siirl is imported (see siirl/__init__.py)
        # All Ray actors inherit this configuration as they import siirl modules.
        rollout_manager = TestRolloutManager.remote(config)
        rollout_workers = ray.get(rollout_manager.get_workers.remote())
        trainer_group = TestTrainerGroup(config, actor_resources,TestTrainer,None,rollout_manager,None)
        trainer_group.init_actors()
        #call sglang engine to generate text 
        sampling_params = {"temperature": 0, "max_new_tokens": 16,"top_k":-1,"top_p":1}
        prompt = "Hello world"
        output_before = ray.get(rollout_workers[0].generate_from_text.remote(prompt, sampling_params))
        text_before = output_before[0]
        logger.info(f"Output before sync: {text_before}")
        
        #update weight
        trainer_group.put_weight()

        #call sglang engine again to generate text and compare with pervious text
        prompt = "Hello world"
        output_after = ray.get(rollout_workers[0].generate_from_text.remote(prompt, sampling_params))
        text_after = output_after[0]
        logger.info(f"Output after sync: {text_after}")

        # compare result
        assert text_before == text_after
        logger.info("Test passed!")
        time.sleep(10)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--rollout_tp", type=int, default=2)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    ray.init(num_cpus=None)
    config = create_grpo_config(tp=args.tp, pp=args.pp,rollout_tp=args.rollout_tp,model_path=args.model_path)
    main_runner = MainRunner.remote()
    ray.get(main_runner.run.remote(config))
    ray.shutdown()
