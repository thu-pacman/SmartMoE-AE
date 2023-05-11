#!/bin/bash

export TOT_EXPERTS=32
export GATE="naive"
export GSHARD_CAP="4.8"

if [ ${GATE} == 'gshard' ];then
    GATE_NAME=${GATE}${GSHARD_CAP}
else
    GATE_NAME=${GATE}
fi
export sparse_name=${TOT_EXPERTS}MoE\_${GATE_NAME}