#!/bin/bash

CONFIG_DIR="conf"
CKPT_DIR="/data2/yoongi/vrvq_github"

DATA_DIR="/path/to/dataset"
SAVE_RESULT_DIR="path/to/save_result_dir"


EXPNAME=${1} # ex) vrvq/vrvq_a2
GPU=${2} #


## Single GPU
run_training_single() {
    # CMD="python scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} ${RESUME_FLAG}"
    CMD="CUDA_VISIBLE_DEVICES=${GPU} python scripts/inference.py \
    --args.load ${CONFIG_DIR}/${EXPNAME}.yml \
    --ckpt_dir ${CKPT_DIR} \
    --tag latest \
    --save_result_dir ${SAVE_RESULT_DIR} \
    --data_dir ${DATA_DIR} \
    --device cuda"

    echo "Running Single GPU: $CMD"
    eval "$CMD"
}

run_training_single
