#!/bin/bash

source $AEROOT/src/env.sh

name=fig8_$(date -Iseconds)

config_path=`pwd`

cd $AEROOT/scripts/GPT

./run_moe.sh $name $config_path/config-fastmoe
mkdir -p $config_path/fastmoe
mv ./logs/$name/* $config_path/fastmoe/.

./run_moe.sh $name $config_path/config-fastermoe
mkdir -p $config_path/fastermoe
mv ./logs/$name/* $config_path/fastermoe/.

./run_moe.sh $name $config_path/config-smartmoe
mkdir -p $config_path/smartmoe
mv ./logs/$name/* $config_path/smartmoe/.

cd -

cd $AEROOT/scripts/deepspeed

./run_deespeed-moe.sh nico 16 1536 32 256 1 2 "nico[1-2]" gshard 1.2
mkdir -p $config_path/deepspeed
mv ./logs/* $config_path/deepspeed/.

cd -
