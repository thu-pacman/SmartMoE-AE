#!/bin/bash

source $AEROOT/src/env.sh

name=fig13_$(date -Iseconds)

bench_path=$AEROOT/scripts/benchmarks
log_dir=$bench_path/logs/$name

cd $bench_path
./run_dist_smart_exchange.sh $name 1 nico1 $AEROOT/src/fastmoe/tests/test_smart_exchange.py $AEROOT/moe_trace/table 8 1536 0 10
./run_dist_smart_exchange.sh $name 1 nico1 $AEROOT/src/fastmoe/tests/test_smart_exchange.py $AEROOT/moe_trace/table 8 1536 0 100

cd -

python3 ./post_process.py $log_dir
