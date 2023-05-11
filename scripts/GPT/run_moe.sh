#!/bin/bash

source $AEROOT/src/env.sh

set -x

if [ "$#" -ne 2]
then
    echo "usage:" $0 "exp_name config_dir"
    exit 1
fi

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

export EXP_NAME=$1
export DENSE_CONFIG=$2/dense.sh
export SPARSE_CONFIG=$2/sparse.sh
export PARALLEL_CONFIG=$2/parallel.sh
export CLUSTER_CONFIG=$2/cluster.sh

. $DENSE_CONFIG
. $SPARSE_CONFIG
. $PARALLEL_CONFIG
. $CLUSTER_CONFIG

export NUM_EXPERTS=$(( $TOT_EXPERTS / $EXPERT_EP_SIZE ))

mkdir -p ./logs
mkdir -p ./logs/${EXP_NAME}
export LOG_DIR=$(pwd)/logs/${EXP_NAME}

LOG_PREFIX=${dense_name}\_${sparse_name}\_${parallel_name}\_${cluster_name}\_$(date -Iseconds)
LOG_NAME=${LOG_PREFIX}.log

export PROFILER_LOG_PATH=${LOG_DIR}/${LOG_PREFIX}.prof

mkdir -p $PROFILER_LOG_PATH

export SCHEDULER_EXEC='srun'
export GPUS_PER_NODE=8

exec_args=""

exec_args+=" --exclusive"
exec_args+=" --export=ALL"
exec_args+=" -K"
exec_args+=" --ntasks-per-node=${GPUS_PER_NODE}"
exec_args+=" --gres=gpu:${GPUS_PER_NODE}"

exec_args+=" -N $NNODES"

if [ $NODELIST != "None" ];then
    tmp=$(scontrol show hostnames ${NODELIST} | wc -l)
    if [ $tmp != $NNODES ];then
        echo "bad nodelist" $NNODES $NODELIST
        exit 1
    fi
    exec_args+=" -w $NODELIST"
fi

exec_args+=" -p AE"

LOG_FILE=${LOG_DIR}/${LOG_NAME}

echo $SCHEDULER_EXEC $exec_args | tee -a ${LOG_FILE}
cat $DENSE_CONFIG | tee -a ${LOG_FILE}
cat $SPARSE_CONFIG | tee -a ${LOG_FILE}
cat $PARALLEL_CONFIG | tee -a ${LOG_FILE}
cat $CLUSTER_CONFIG | tee -a ${LOG_FILE}

$(which $SCHEDULER_EXEC) \
    $exec_args \
	./pretrain_moe.sh \
	2>&1 | tee -a ${LOG_FILE}
