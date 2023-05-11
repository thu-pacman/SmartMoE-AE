#!/bin/bash

prefix=$1

cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'

cat $prefix.prof/16.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/16.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/16.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/16.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'

cat $prefix.prof/32.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/32.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/32.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/32.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'

cat $prefix.prof/48.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/48.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/48.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/48.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
