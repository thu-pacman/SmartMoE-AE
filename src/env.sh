#!/bin/bash

source /opt/spack/share/spack/setup-env.sh 
spack load gcc@9.2.0
spack load nccl@2.10.3-1/6qeub7l 
spack load cudnn/dvbx
