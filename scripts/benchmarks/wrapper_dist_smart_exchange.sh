#! /bin/bash

set -x

export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NPROCS
localrank=$SLURM_LOCALID

export NODE_RANK=$(( $RANK / $NNODES))
export CUDA_VISIBLE_DEVICES=$localrank

exec python3 $EXEC $TABLE_PREFIX $MICRO_BATCH_SIZE $HIDDEN_SIZE $HISTORY_LAT $UPDATE_FREQ
