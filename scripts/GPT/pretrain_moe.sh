#! /bin/bash

set -x

export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NPROCS
localrank=$SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=$localrank
export NODE_RANK=$(( $RANK / $NNODES))

# ninja compiler jobs
export MAX_JOBS=64

cd ${CODE_PREFIX}

python_args="
        --fmoefy \
        --num-experts $NUM_EXPERTS \
        --balance-strategy $GATE \
        --gshard-cap ${GSHARD_CAP} \
        --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
        --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --train-samples $TRAIN_SAMPLES \
		--lr-decay-samples 4882800 \
        --lr 0.0001 \
        --min-lr 0.00001 \
        --lr-decay-style cosine \
        --initial-loss-scale 131072 \
        --log-interval 1 \
        --eval-iters -1 \
        --data-path ${DATA_PATH} \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --fp16 \
        --DDP-impl local \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --expert-ep-size ${EXPERT_EP_SIZE} \
        --expert-dp-size ${EXPERT_DP_SIZE} \
        --vocab-file $GPT_VOCAB_FILE \
        --merge-file $MERGE_FILE "

if [ ${DYNAMIC_ENABLE} == "ON" ];then
    python_args+="  --dynamic-placement \
                    --dynamic-freq ${DYNAMIC_FREQ} "
fi

EXEC="./pretrain_gpt.py"

echo $EXEC $python_args

USE_MEGATRON=1 \
PROFILER_LOG_PATH=$PROFILER_LOG_PATH \
exec python3 \
        $EXEC \
        $python_args
