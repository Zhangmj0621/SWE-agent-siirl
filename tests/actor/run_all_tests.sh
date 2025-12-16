# Usage:
#   bash run_all_tests.sh init    # Initialize baseline only
#   bash run_all_tests.sh         # Run all tests (default)

#!/bin/bash
set -e

MODEL_PATH="/inspire/hdd/global_user/wangzhixin-240108090034/model/hf_qwen3_0.6b"

MODE="${1:-test}"

# Baseline initialization
if [ "$MODE" == "init" ]; then
    echo "Initializing baseline..."
    torchrun --nproc_per_node=1 test_actor_training.py --init --model-path "$MODEL_PATH"
    exit 0
fi

# 2-GPU tests
echo "Running 2-GPU tests..."
for config in tp2 pp2 cp2 dp2 tp2_sp; do
    echo "Testing: $config"
    torchrun --nproc_per_node=2 test_actor_training.py --config $config --model-path "$MODEL_PATH"
done

# 4-GPU tests
echo "Running 4-GPU tests..."
for config in tp4 pp4 dp4 tp2_pp2 tp2_cp2 tp2_dp2 pp2_dp2 cp2_dp2 tp4_sp tp2_dp2_sp tp2_pp2_sp; do
    echo "Testing: $config"
    torchrun --nproc_per_node=4 test_actor_training.py --config $config --model-path "$MODEL_PATH"
done

# 8-GPU tests
echo "Running 8-GPU tests..."
for config in tp8 pp8 dp8 tp2_pp4 tp4_pp2 tp2_dp4 tp4_dp2 pp2_dp4 pp4_dp2 cp2_dp4 tp2_pp2_dp2 tp2_pp2_cp2; do
    echo "Testing: $config"
    torchrun --nproc_per_node=8 test_actor_training.py --config $config --model-path "$MODEL_PATH"
done

echo "All tests completed!"
