#!/bin/bash

export TOT_EXPERTS=32
export GATE="gshard"
export GSHARD_CAP="1.2"

if [ ${GATE} == 'gshard' ];then
    GATE_NAME=${GATE}${GSHARD_CAP}
else
    GATE_NAME=${GATE}
fi
export sparse_name=${TOT_EXPERTS}MoE\_${GATE_NAME}