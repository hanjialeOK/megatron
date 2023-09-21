#!/bin/bash

cd $(dirname ${BASH_SOURCE[0]})/..

cd megatron/data/

echo "Compile helpers in every worker..."
make
cd ../../

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
# ARNOLD_WORKER_0_HOST METIS_WORKER_0_HOST DMLC_PS_ROOT_URI
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
MASTER_PORT=$ARNOLD_WORKER_0_PORT
# ARNOLD_NUM METIS_TASK_REPLICA ARNOLD_WORKER_NUM DMLC_NUM_WORKER BYTEPS_NUM_NODES
NNODES=${NNODES:=$ARNOLD_WORKER_NUM}
NODE_RANK=$ARNOLD_ID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MBS=${MBS:=4}
GBS=${GBS:=32}
NLS=${NLS:=24}
HS=${HS:=1024}
NAH=${NAH:=16}
ITERS=${ITERS:=1000000}
DITERS=${DITERS:=990000}

CHECKPOINT_PATH=checkpoint/
VOCAB_FILE=data/bert-large-cased-vocab.txt
DATA_PATH=data/data-small3_text_sentence

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BERT_ARGS="
    --num-layers $NLS \
    --hidden-size $HS \
    --num-attention-heads $NAH \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.0001 \
    --train-iters $ITERS \
    --lr-decay-iters $DITERS \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
