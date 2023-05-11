#!/bin/bash

set -x

if [ "$#" -ne 9 ]
then
    echo "usage:" $0 "exp_name nnodes nodelist exec table_prefix mbs hidden_size history_lat update_freq"
    exit 1
fi

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

export EXP_NAME=$1
export NNODES=$2
export NODELIST=$3
export EXEC=$4
export TABLE_PREFIX=$5
export MICRO_BATCH_SIZE=$6
export HIDDEN_SIZE=$7
export HISTORY_LAT=$8
export UPDATE_FREQ=$9

export FMOE_FASTER_GLBPLC_NETBW="8e9"
export FMOE_FASTER_GLBPLC_NETBW_Bcast="2e9"
export FMOE_FASTER_GLBPLC_GPUTP="112e12"

export FMOE_FASTER_SCHEDULE_ENABLE=ON
export FMOE_FASTER_GROUP_SIZE=16
export FMOE_FASTER_SHADOW_ENABLE=ON
export FMOE_FASTER_GLBPLC_ALPHA="2"
export FMOE_FASTER_GLBPLC_DMODEL=${HIDDEN_SIZE}

mkdir -p ./logs
mkdir -p ./logs/${EXP_NAME}

LOG_DIR=$(pwd)/logs/${EXP_NAME}

tmp=${TABLE_PREFIX#/*logs/}
result=$(echo $tmp | sed "s/\//_/g")
LOG_PREFIX=${NNODES}nodes_mbs${MICRO_BATCH_SIZE}\_H${HIDDEN_SIZE}\_LAT${HISTORY_LAT}\_FREQ${UPDATE_FREQ}\_$(date -Iseconds)
LOG_NAME=${LOG_PREFIX}.log

export SCHEDULER_EXEC='srun'
export GPUS_PER_NODE=8

if [ ${NNODES} == "half" ];then
    export NNODES=1
    export GPUS_PER_NODE=4
fi

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

echo $SCHEDULER_EXEC $exec_args

$(which $SCHEDULER_EXEC) \
    $exec_args \
	./wrapper_dist_smart_exchange.sh \
    2>&1 | tee ${LOG_DIR}/${LOG_NAME}
