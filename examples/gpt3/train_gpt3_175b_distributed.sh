#!/bin/bash

# Runs the "175B" parameter model
#* 设置 CUDA 最大连接数，以防止 GPU 之间的过多连接导致性能问题
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
#!  多节点配置参数
MASTER_ADDR=localhost  #? 主节点地址（如果多节点部署，需要更改）
MASTER_PORT=6000       #? 主节点通信端口
NUM_NODES=1            #? 节点数量
NODE_RANK=0            #? 当前节点的排名（单节点训练时为 0）
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))  #? 总 GPU 数量

#!  从脚本输入参数中读取路径信息
CHECKPOINT_PATH=$1          #? 指定模型检查点路径
TENSORBOARD_LOGS_PATH=$2    #? 指定 TensorBoard 日志路径
VOCAB_FILE=$3               #? 指定词汇表文件路径 (GPT-2)
MERGE_FILE=$4               #? 指定合并文件路径 (GPT-2)
DATA_PATH=$5                #? 指定训练数据路径

#!  分布式训练参数
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE  #? 每个节点上的进程数 (GPU数)
    --nnodes $NUM_NODES              #? 总节点数
    --master_addr $MASTER_ADDR       #? 主节点地址
    --master_port $MASTER_PORT       #? 主节点通信端口
)

#!  GPT 模型超参数配置
GPT_MODEL_ARGS=(
    --num-layers 96 
    --hidden-size 12288 
    --num-attention-heads 96 
    --seq-length 2048 
    --max-position-embeddings 2048 
)

#!  训练超参数配置
TRAINING_ARGS=(
    --micro-batch-size 1                #? 微批次大小（每个 GPU 上的批次大小）
    --global-batch-size 1536            #? 全局批次大小
    --rampup-batch-size 16 16 5859375   #! 逐步增加批次大小的计划 [初始批次大小, 最终批次大小, 增加步数]
    --train-iters 500000                #? 训练迭代次数
    --weight-decay 0.1                  #? 权重衰减系数  
    --adam-beta1 0.9                    #? Adam 优化器 beta1 参数
    --adam-beta2 0.95                   #? Adam 优化器 beta2 参数
    --init-method-std 0.006             #? 初始化方法标准差
    --clip-grad 1.0                     #? 梯度裁剪阈值
    --fp16                              #? 混合精度训练
    --lr 6.0e-5                         #? 学习率
    --lr-decay-style cosine             #? 学习率衰减方式 
    --min-lr 6.0e-6                     #? 最小学习率
    --lr-warmup-fraction .001           #? 学习率预热比例
    --lr-decay-iters 430000             #? 学习率衰减的总步数
)

#! 模型并行参数（张量并行与流水线并行）
MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 8 
	--pipeline-model-parallel-size 16 
)

#! 数据相关参数
DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

#! 评估和日志记录参数
EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
