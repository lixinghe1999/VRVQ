#!/bin/bash

CONFIG_DIR="conf"
SAVE_DIR="/data2/yoongi/vrvq_github"

EXPNAME=${1}
GPU=${2}
RESUME=${3}

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume"
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPU"
NUM_GPUS=${#GPU_ARRAY[@]}

## Single GPU
run_training_single() {
    # CMD="python scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} ${RESUME_FLAG}"
    CMD="CUDA_VISIBLE_DEVICES=${GPU} python scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} ${RESUME_FLAG}"
    echo "Running Single GPU: $CMD"
    eval "$CMD"
}

# run_training_single()

## Multi GPU
run_training_multi() {
    # CMD="torchrun --nproc_per_node=${GPU} scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} ${RESUME_FLAG}"
    CMD="CUDA_VISIBLE_DEVICES=${GPU} torchrun --nproc_per_node gpu scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} ${RESUME_FLAG}"
    echo "Running Multi GPU: $CMD"
    eval "$CMD"
}

if [ "$NUM_GPUS" -eq 1 ]; then
    run_training_single
else
    run_training_multi
fi

# run_training_multi()

# ## Single GPU
# if [ -z "${RESUME}" ]; then
#     python scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME}
# else
#     python scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} --resume
# fi

# ## Multi GPU
# if [ -z "${RESUME}" ]; then
#     torchrun --nproc_per_node gpu scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME}
# else
#     torchrun --nproc_per_node gpu scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} --resume
# fi

