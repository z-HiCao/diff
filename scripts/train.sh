NUM_MACHINES=1
NUM_LOCAL_GPUS=1 # 8
MACHINE_RANK=0
MAIN_MACHINE_IP="10.126.62.117"  # fill your machine IP here
MAIN_MACHINE_PROT="25028"  # fill your machine port here

FILE=$1
CONFIG_FILE=$2
TAG=$3
shift 3  # remove $1~$3 for $@

# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export TORCH_HOME=~/.cache/torch
export NCCL_DEBUG=VERSION
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

accelerate launch \
    --config_file /root/.cache/huggingface/accelerate/default_config.yaml \
    --mixed_precision=fp16 \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_MACHINE_IP \
    --main_process_port $MAIN_MACHINE_PROT \
    ${FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --allow_tf32 \
        $@
