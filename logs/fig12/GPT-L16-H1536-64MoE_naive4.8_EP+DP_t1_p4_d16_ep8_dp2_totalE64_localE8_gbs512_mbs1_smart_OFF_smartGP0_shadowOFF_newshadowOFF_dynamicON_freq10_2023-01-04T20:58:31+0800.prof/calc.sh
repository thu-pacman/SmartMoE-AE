#!/bin/bash

cat 0.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 0.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 0.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 0.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'

cat 16.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 16.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 16.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 16.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'

cat 32.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 32.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 32.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 32.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'

cat 48.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 48.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 48.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat 48.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
