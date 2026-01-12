DP=1
TP=2
PP=2
GPUS_PER_NODE=$(($TP * $PP* $DP))
torchrun --nproc_per_node $GPUS_PER_NODE \
    --master_port 29501 \
    /inspire/hdd/project/qianghuaxuexi/wangtongyu-25057/wty_public/siirl-agentic/tests/actor/test_mbridge.py
