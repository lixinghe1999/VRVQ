#!/bin/bash


## ex for multi_gpu) in VRVQ/, bash scripts/script_train.sh vrvq/vrvq_a2 2,3
CONFIG_DIR="conf"
SAVE_DIR="/data2/yoongi/vrvq_github"

EXPNAME=${1}
GPU=${2}

## Single GPU
run_training_single() {
    # CMD="python scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} ${RESUME_FLAG}"
    CMD="CUDA_VISIBLE_DEVICES=${GPU} python scripts/inference.py \
    --args.load ${CONFIG_DIR}/${EXPNAME}.yml \
    --ckpt_dir ${SAVE_DIR} \
    --tag latest \
    --save_result_dir results \
    --device cuda"

    echo "Running Single GPU: $CMD"
    eval "$CMD"
}

run_training_single
