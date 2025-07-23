CONFIG_DIR="conf"
SAVE_DIR="./runs/24kbps"
EXPNAME=${1}
GPU=${2}
RESUME=${3}
PORT=29550
RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume"
fi
IFS=',' read -ra GPU_ARRAY <<< "$GPU"
NUM_GPUS=${#GPU_ARRAY[@]}
echo "Running Multi GPU"
CMD="CUDA_VISIBLE_DEVICES=${GPU} MASTER_ADDR=localhost MASTER_PORT=$PORT python -m torch.distributed.run --nproc_per_node gpu --master_port=$PORT scripts/train.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} ${RESUME_FLAG}"
eval "$CMD"