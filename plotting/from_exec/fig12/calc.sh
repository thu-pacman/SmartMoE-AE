#!/bin/bash

prefix=$1

cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L4_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L5_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L6_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/0.profiler | sed "s/,/\n/g" | grep "MoE_L7_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'

cat $prefix.prof/8.profiler | sed "s/,/\n/g" | grep "MoE_L0_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/8.profiler | sed "s/,/\n/g" | grep "MoE_L1_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/8.profiler | sed "s/,/\n/g" | grep "MoE_L2_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/8.profiler | sed "s/,/\n/g" | grep "MoE_L3_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/8.profiler | sed "s/,/\n/g" | grep "MoE_L4_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/8.profiler | sed "s/,/\n/g" | grep "MoE_L5_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/8.profiler | sed "s/,/\n/g" | grep "MoE_L6_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
cat $prefix.prof/8.profiler | sed "s/,/\n/g" | grep "MoE_L7_[b]wd" | awk 'BEGIN{cnt=0}{cnt+=$2}END{print cnt/1000.0}'
