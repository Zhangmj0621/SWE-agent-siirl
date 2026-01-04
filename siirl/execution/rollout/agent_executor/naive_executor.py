# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
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
import os
import copy
import asyncio
import ray
import time
import torch
import importlib
import numpy as np

from tqdm.asyncio import tqdm_asyncio
from collections import deque
from loguru import logger
from typing import List, Set, Dict, Any, Tuple
from siirl.params.training_args import SiiRLArguments
from siirl.data_coordinator.sample import Sample, SampleInfo, Samples2Dict
from siirl.utils.timer import Timer
from siirl.utils.model_utils.model import compute_position_id_with_mask
class NaiveExecutor:
    '''
    NaiveExecutor used in synchronous training workflows.
    Manages asynchronous sample generation, pre/post processing, and data coordination
    for rollout processes in reinforcement learning with large language models.
    '''
    def __init__(self, config:SiiRLArguments, data_coordinator, engine, train_batch_size):
        """
        Initialize NaiveExecutor with core configuration and dependencies.
        
        Args:
            config: SiiRLArguments containing all training/rollout hyperparameters
            data_coordinator: Ray handle to data coordinator for sample management
            engine: Inference engine instance (e.g., SglangEngine) for text generation
            train_batch_size: Batch size for rollout sample generation
        """
        self.config = config
        self.data_coordinator = data_coordinator  # Ray actor handle to data coordinator
        self.running = False  # Flag to control executor main loop
        self.engine = engine  # Inference engine for text generation
        # TODO: use validation arguments here
        self.train_batch_size = train_batch_size # Target batch size for rollout samples
        self.max_concurrency_size = train_batch_size * config.rollout.n
        self.tasks:Set[asyncio.Task] = set()  # Track active generation tasks for cleanup
        self.finish_group_samples:Dict[str, List[Any]] = {} # Save result of finish samples until reach n group
        self.pending_queue = deque()
        self.rollout_n = config.rollout.n

        # Semaphore to control concurrent generation tasks (limit to batch size)
        self.semaphore = asyncio.Semaphore(self.max_concurrency_size) 
        self.reward_fn = None  # Custom reward function (optional)
        self.rollout_flow = None  # Rollout flow function for sample generation
        self._rank = int(os.environ.get('RANK'))
        # Load custom reward function if configured
        if config.custom_reward_function.path:
            from siirl.utils.reward_score.custom_reward import load_custom_reward_function
            self.reward_fn = load_custom_reward_function(config=config)
        
        # Load rollout flow function (naive or custom)
        flow_path = config.rollout.flow_function
        if flow_path == "naive":
            from siirl.execution.rollout.agent_flow.naive_flow import NaiveFlow
            self.rollout_flow = NaiveFlow(self.config, self.engine)
        elif flow_path == "agent":
            from siirl.execution.rollout.agent_flow.agent_flow import build_agentflow
            import yaml

            with open(config.rollout.flow_config, "r") as f:
                flow_config = yaml.safe_load(f)
            self.rollout_flow = build_agentflow(flow_config, engine)
        else:
            # Dynamically import custom rollout flow function
            module_path, name = flow_path.rsplit('.', 1)
            mod = importlib.import_module(module_path)
            self.rollout_flow = getattr(mod, name)

    async def init_sample(self):
        need_replenish = self.max_concurrency_size

    

    async def get_sample(self):
        """
        Get new samples from data coordinator to replenish the batch.
        Calculates the number of missing samples and requests them from data coordinator.
        
        Returns:
            List of new samples from data coordinator (empty if batch is full)
        """
        # Calculate number of samples needed to reach target batch size
        need_replenish = self.max_concurrency_size - len(self.tasks)
        if need_replenish == 0:
            return []
        # Request new samples from data coordinator (Ray remote call)
        if len(self.pending_queue) < need_replenish:
            diff = need_replenish - len(self.pending_queue)
            pull_size = (diff + self.rollout_n - 1) // self.rollout_n
            
            pull_samples = await self.data_coordinator.get_dataloader.remote(pull_size)
            
            for sample in pull_samples:
                samples = [copy.deepcopy(sample) for _ in range(self.rollout_n)]
                self.pending_queue.extend(samples)
        new_samples = []

        for _ in range(need_replenish): 
            if len(self.pending_queue):
                new_samples.append(self.pending_queue.popleft())
        return new_samples
    
    
    def _manual_pad(self, ids, max_length, padding_side="right"):
        """
        Manually pad token IDs and create corresponding attention mask.
        Handles both left and right padding for sequence length standardization.
        
        Args:
            ids: 2D list of token IDs (batch_size=1, [List[int]])
            max_length: Target sequence length after padding
            padding_side: Direction for padding ("left" or "right", default: "right")
        
        Returns:
            Tuple of (padded_ids, attention_mask) as torch tensors (shape: [1, max_length])
        """
        # Convert list to tensor (batch_size=1, seq_len)
        ids = torch.tensor(ids, dtype=torch.long)
        pad_length = max_length - ids.shape[1]
        
        # No padding needed - truncate to max length and create all-ones attention mask
        if pad_length <= 0:
            padded_ids = ids[:, :max_length]
            attention_mask = torch.ones_like(padded_ids, dtype=torch.long)
            return padded_ids, attention_mask
        
        # Create padding tensor with pad token ID
        pad_tensor = torch.full((ids.shape[0], pad_length), self.engine.tokenizer.pad_token_id, dtype=torch.long)
        
        # Left padding: pad first, then original sequence
        if padding_side == "left":
            padded_ids = torch.cat([pad_tensor, ids], dim=1)
            attention_mask = torch.cat([torch.zeros_like(pad_tensor), torch.ones_like(ids)], dim=1)
        # Right padding: original sequence first, then pad
        else:  # right
            padded_ids = torch.cat([ids, pad_tensor], dim=1)
            attention_mask = torch.cat([torch.ones_like(ids), torch.zeros_like(pad_tensor)], dim=1)
        
        return padded_ids, attention_mask

    def _pre_process(self, sample:Sample, is_validate = False):
        """
        Preprocess single sample before generation.
        Maps raw prompt IDs to prompts field for consistency.
        
        Args:
            sample: Sample object to preprocess
        
        Returns:
            Preprocessed sample with prompts field set
        """
        sample.prompts = sample.raw_prompt_ids
        if is_validate:
            sample.prompt_texts = self.engine.tokenizer.decode(sample.prompts)
        return sample
    
    def _post_process(self, sample:Sample):
        """
        Postprocess generated sample with padding, sequence concatenation, and reward formatting.
        Standardizes prompt/response lengths, creates attention masks, and formats reward tensors.
        
        Args:
            sample: Generated sample to postprocess
        
        Returns:
            Postprocessed sample with standardized tensor fields
        """
        # Pad prompts to max prompt length (left padding for prompt sequences)
        prompt_ids, prompt_attention_mask = self._manual_pad(
            [sample.prompts],
            max_length=self.config.data.max_prompt_length,
            padding_side="left"
        )

        # Pad responses to max response length (right padding for generated text)
        response_ids, response_attention_mask = self._manual_pad(
            [sample.responses],
            max_length=self.config.data.max_response_length,
            padding_side="right"
        )

        # Pad and combine response mask with attention mask (filter padding tokens)
        response_mask, _ = self._manual_pad(
            [sample.response_mask],
            max_length=self.config.data.max_response_length,
            padding_side="right"
        )
        response_mask = response_mask * response_attention_mask
        
        # Pad rollout_log_prob to match response length (for monitoring metrics)
        if sample.rollout_log_prob is not None:
            max_len = self.config.data.max_response_length
            current_len = len(sample.rollout_log_prob)
            if current_len < max_len:
                pad_len = max_len - current_len
                sample.rollout_log_prob = np.pad(
                    sample.rollout_log_prob, (0, pad_len), 
                    mode='constant', constant_values=0.0
                ).astype(np.float32)
            elif current_len > max_len:
                sample.rollout_log_prob = sample.rollout_log_prob[:max_len].astype(np.float32)
        
        # Validate tensor shape consistency
        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )

        # Concatenate prompt and response sequences for model input
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        
        # Create position IDs (account for padding in attention mask)
        # Initialize position_ids with the same shape as attention_mask
        # position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
        # # Get the attention mask for this sample
        # valid_indices = attention_mask [0].nonzero(as_tuple=False).squeeze(-1)
        # seq_length = attention_mask.size(1)
        # if len(valid_indices) > 0:
        #     first_valid = valid_indices[0].item()
        #     # Left padding positions remain 0
        #     # Valid positions get incremental IDs starting from 0
        #     # Right padding positions continue the sequence

        #     # Assign position IDs for the entire sequence starting from first valid position
        #     position_sequence = torch.arange(seq_length - first_valid)
        #     position_ids[0, first_valid:] = position_sequence
        
        # position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Format reward tensor (place reward value at last valid response token position)
        reward_tensor = torch.zeros_like(response_ids[0], dtype=torch.float32)
        prompt_length = prompt_ids[0].shape[-1]
        valid_response_length = attention_mask[0][prompt_length:].sum()
        reward_tensor[valid_response_length - 1] = sample.rewards
        # Clean up and set processed fields in sample
        sample.token_level_rewards = reward_tensor.numpy()
        sample.token_level_scores = copy.deepcopy(reward_tensor.numpy())
        sample.prompts = prompt_ids[0].numpy()
        sample.responses = response_ids[0].numpy()
        sample.response_mask = response_mask[0].numpy()
        sample.input_ids = input_ids[0].numpy()
        sample.attention_mask = attention_mask[0].numpy()
        sample.position_ids = position_ids[0].numpy()
        
        return sample   
    
    async def put_data(self, sample, loop):
        sample_ref = await loop.run_in_executor(
                    None, 
                    ray.put, 
                    sample
                )
        
        # Create sample metadata for tracking
        sample_info = SampleInfo(
            sum_tokens=getattr(sample, 'sum_tokens', int(sample.attention_mask.sum())),
            prompt_length=getattr(sample, 'prompt_length', 0),
            response_length=getattr(sample, 'response_length', 0),
            uid=str(sample.uid),
            weight_version=self.engine.weights_version,
            dict_info={
                'key': "Actor",
            })
        if sample.uid not in self.finish_group_samples:
            self.finish_group_samples[sample.uid] = []
        self.finish_group_samples[sample.uid].append((sample_info, sample_ref))
        # Send processed sample to data coordinator
        if len(self.finish_group_samples[sample.uid]) == self.rollout_n:
            tuple_datas = self.finish_group_samples.pop(sample.uid)
            sample_infos = [tuple_data[0] for tuple_data in tuple_datas]
            sample_refs = [tuple_data[1] for tuple_data in tuple_datas]
            await self.data_coordinator.put_batch.remote(sample_infos, sample_refs)

        return sample
    
    async def generate(self, sample, is_validate = False):
        """
        Asynchronous sample generation pipeline: preprocess → rollout → postprocess → data coordination.
        Uses semaphore to control concurrency and offloads CPU-bound processing to executor.
        
        Args:
            sample: Raw sample from data coordinator
        
        Returns:
            Postprocessed sample with generated response and formatted tensors, or None if failed
        """
        # Record timing information for performance analysis
        timing_info = {
            "rollout_start_at": time.time(),
        }
        sample_uid = getattr(sample, 'uid', 'unknown')
        
        try:
            async with self.semaphore:  # Limit concurrent generations to batch size
                loop = asyncio.get_running_loop()
                # 1. Preprocess sample (CPU-bound, offload to executor)
                sample = await loop.run_in_executor(
                            None, 
                            self._pre_process, 
                            sample, is_validate
                        )
                
                # 2. Execute rollout flow (LLM generation with reward calculation)
                sample = await self.rollout_flow(sample, self.reward_fn, is_validate)
                
                # 3. Postprocess sample (CPU-bound padding and tensor formatting)
                sample = await loop.run_in_executor(
                            None, 
                            self._post_process, 
                            sample
                        )
                
                # 4. Collect timing information from rollout flow
                timing_info["rollout_end_at"] = time.time()
                timing_info["rollout_duration"] = timing_info["rollout_end_at"] - timing_info["rollout_start_at"]
                timing_info["generation_duration"] = getattr(sample, "_generation_duration", 0)
                timing_info["reward_duration"] = getattr(sample, "_reward_duration", 0)
                sample.timing_info = timing_info
                
                # 5. Store processed sample in Ray object store and notify data coordinator
                if not is_validate:
                    await self.put_data(sample = sample, loop = loop)
                return sample
                
        except Exception as e:
            import traceback
            logger.error(f"[NaiveExecutor.generate] Sample uid={sample_uid} failed: {e}")
            logger.error(f"[NaiveExecutor.generate] Traceback:\n{traceback.format_exc()}")
            raise

    async def run(self):
        """
        Main execution loop for the executor.
        Continuously replenishes samples, creates generation tasks, and maintains batch size.
        Runs until self.running is set to False.
        """
        self.running = True
        stats_task = None
        if self._rank == 0:
            stats_task = asyncio.create_task(self.rollout_status())
        
        while self.running:
            # Get new samples to replenish batch
            samples = await self.get_sample()
            if not samples:
                # No new samples - short sleep to avoid busy waiting
                await asyncio.sleep(0.001)
            else:
                # Create generation tasks for new samples
                tasks = []
                for sample in samples:
                    task = asyncio.create_task(self.generate(sample))
                    tasks.append(task)
                    self.tasks.add(task)
                    # Remove task from tracking set when completed
                    task.add_done_callback(self.tasks.discard)
                
                # Debug code (commented out) - save batch for inspection
                # samples = await asyncio.gather(*tasks)
                # batch = Samples2Dict(samples=samples)
                # torch.save(batch, f"save_dict/{os.environ.get('RANK')}_batch.pt")
                # break
                
                # Yield control to event loop (non-blocking sleep)
                await asyncio.sleep(0)  
        if self._rank == 0:
            stats_task.cancel()
            await asyncio.gather(stats_task, return_exceptions=True)
     
    async def validate(self, val_batch_size) -> Tuple[List[Sample],Dict]:
        val_samples = []
        with Timer("get_val_data") as val_get_time:
            while True:
                data = ray.get(self.data_coordinator.get_dataloader.remote(batch_size = val_batch_size, is_validate = True))
                if len(val_samples) and not len(data):
                    break
                val_samples.extend(data)
        val_tasks = []
        logger.info(f"RANK_{self._rank} start validate, batch_size:{len(val_samples)}")
        with Timer("val_generate") as val_generate_time:
            for sample in val_samples:
                task = asyncio.create_task(self.generate(sample, is_validate = True))
                val_tasks.append(task)
            result = await tqdm_asyncio.gather(
                    *val_tasks,
                    desc=f"[Rank_{self._rank}]-Validate",
                    unit="sample",
                    dynamic_ncols=True,
                    mininterval=2.0,     
                    miniters=50,         
                )
        metrics = {"val_get_time":val_get_time.elapsed, "val_generate_time":val_generate_time.elapsed}
        return result, metrics
       
    async def rollout_status(self, interval: float = 10.0):
        last_status = 0
        while True:
            await asyncio.sleep(interval)
            current_status = len(self.tasks)
            if last_status != current_status :
                logger.info(f"rank_{self._rank} active generate tasks: {current_status}, {len(self.pending_queue)} left in pending_queue")
                last_status = current_status
          
    async def stop(self):
        """
        Stop executor and clean up active tasks.
        Sets running flag to False and waits for all active generation tasks to complete.
        """
        self.running = False
        # Wait for all remaining tasks to finish before exiting
        await asyncio.gather(*self.tasks)