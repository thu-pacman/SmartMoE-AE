#!/bin/bash

export AEROOT=`pwd`

output_dir=outputs_from_log_$(date -Iseconds)
mkdir $output_dir

python3 ./plotting/from_logs/fig8.py $AEROOT $output_dir
python3 ./plotting/from_logs/fig9.py $AEROOT $output_dir
python3 ./plotting/from_logs/fig10.py $AEROOT $output_dir
python3 ./plotting/from_logs/fig11.py $AEROOT $output_dir
python3 ./plotting/from_logs/fig12.py $AEROOT $output_dir
python3 ./plotting/from_logs/fig13.py $AEROOT $output_dir