#!/bin/bash

if [ ${SCHEDULER_EXEC} == 'srun' ];then
    export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NPROCS
    localrank=$SLURM_LOCALID
else
    RANK=0
    localrank=0
    WORLD_SIZE=1
fi


export NODE_RANK=$(( $RANK / $NNODES))
export LOCAL_RANK=$localrank

# ninja compiler jobs
export MAX_JOBS=64

CODE_PREFIX=$AEROOT/src

cd ${CODE_PREFIX}/Megatron-DeepSpeed

exec python3 $run_cmd
