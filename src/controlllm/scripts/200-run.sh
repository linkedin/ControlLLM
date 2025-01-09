#!/bin/sh
export PATH=$HOME/.local/bin:$PATH
# source scl_source enable devtoolset-11
# source mldev-scripts/setup_mlflow_hf.sh
# dragon knight

export GPUS_PER_NODE=$(nvidia-smi --list-gpus|wc -l)
export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
# Note that to make qgZ work(zero_quantized_gradients): only magic numbers will work, one of [1, 2, 3, 5, 8, 16, 32, 40, 64, 80 ...]
export NUM_NODES=1  # 1 means 1 node, set to 2 for 2 nodes.
export WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# this is to enforce the usage of the offline datasets from huggingface
export HF_DATASETS_OFFLINE=1
# this is to make sure nccl does not timeout
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=36000
# this is to memory Fragmentation over time following https://github.com/pytorch/pytorch/issues/130330
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # this is to enable wandb, not in use for now
# export WANDB_API_KEY=your-api-key

# choose the trainer to use
export TRAINER=native  # or transformers/native

OUTPUT_DIR="$FLYTE_INTERNAL_EXECUTION_ID-$(date -Iseconds)"

# native torch trainer:
torchrun --nnodes=$NUM_NODES --nproc-per-node=$LOCAL_WORLD_SIZE \
      --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_id=1234 --rdzv_backend=c10d \
      /home/jobuser/controlllm/main.py
# #       --dataset JobseekerDataset \
# #       --enable_fsdp


# # transformer trainer:
# torchrun --nnodes=$NUM_NODES --nproc-per-node=$LOCAL_WORLD_SIZE \
#       --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_id=1234 --rdzv_backend=c10d \
#       /home/jobuser/controlllm/main.py
# #       --dataset JobseekerDatasetFull \
# #       --enable_fsdp false  # go to ./controlllm/config/trainer_config.py to disable enable_fsdp
# #       --enable_deepspeed
# #       --deepspeed /home/jobuser/controlllm/configs/z3_++.json
