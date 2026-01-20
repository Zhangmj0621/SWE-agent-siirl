#!/bin/bash
set -x

# 设置 PYTHONPATH 以包含项目依赖
export PYTHONPATH=/inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/wty_public/siirl-agentic:/inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/wty_public/workspace/Megatron-LM_siirl:$PYTHONPATH

# 设置 Megatron 运行所需的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 并行配置 (Tensor Parallel 和 Pipeline Parallel)
DP=1
TP=4
PP=1
GPUS_PER_NODE=$(($TP * $PP* $DP))
# export tensor_compare=1
# export dump_tensor_data_path="/inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/wty_public/saved_tensor_dict/dump_tensors_01"
# export input_with_dump_data="/inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/wty_public/saved_tensor_dict/full_info/grpo_qwen3_0.6b_tensor_dict"
# 使用 torchrun 启动分布式测试
torchrun --nproc_per_node $GPUS_PER_NODE \
    --master_port 29501 \
    /inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/wty_public/siirl-agentic/tests/actor/siirl_test_actor.py \
    --tp $TP \
    --pp $PP \
    --model-path "/inspire/ssd/project/qianghuaxuexi/public/debug_models/Qwen3-0.6B" \
    --tensordict-data-path "/inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/wty_public/saved_tensor_dict/full_info/ppo_qwen3_0.6b_tensor_dict_mbridge" \
    --algo ppo
