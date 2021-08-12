#!/usr/bin/env bash
#example: nohup sh tools/dist_train.sh configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_boephone_rgb.py 4 > 1.log &
#

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
export CUDA_VISIBLE_DEVICES="1,2,3"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}
