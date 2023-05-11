#!/bin/bash

source $AEROOT/src/env.sh

name=fig12_$(date -Iseconds)

config_path=`pwd`

cd $AEROOT/scripts/GPT

./run_moe.sh $name $config_path/config-static
./run_moe.sh $name $config_path/config-dynamic

cd -

cp -r $AEROOT/scripts/GPT/logs/$name/* .
