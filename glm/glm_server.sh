#! /bin/bash

CUDA_LAUNCH_BLOCKING=1

if [ x"${MPSIZE}" == "x" ]; then
    MPSIZE=$(ls $CHECKPOINT_PATH | grep mp_rank_ | sort -r | head -n 1 | tr -cd '[0-9]')
    MPSIZE=$(expr $MPSIZE + 1)
fi

MAXSEQLEN=3000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

DISTRIBUTED_ARGS="--nproc_per_node $MPSIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS $script_dir/glm_server.py \
       --world_size $MPSIZE \
       --tensor_para_size $MPSIZE \
       --pipeline_para_size 1 \
       --max_seq_len $MAXSEQLEN \
       --ckpt_path $CHECKPOINT_PATH \
       --data_type $DATA_TYPE
