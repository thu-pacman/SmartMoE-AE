#!/bin/bash

source $AEROOT/src/env.sh

name=fig10_$(date -Iseconds)

config_path=`pwd`

cd $AEROOT/scripts/GPT

./run_moe.sh $name $config_path/config-fastmoe
mkdir -p $config_path/fastmoe
mv ./logs/$name/* $config_path/fastmoe/.

./run_moe.sh $name $config_path/config-alpa
mkdir -p $config_path/alpa
mv ./logs/$name/* $config_path/alpa/.

./run_moe.sh $name $config_path/config-smartmoe
mkdir -p $config_path/smartmoe
mv ./logs/$name/* $config_path/smartmoe/.

cd -
