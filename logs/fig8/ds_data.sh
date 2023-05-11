#!/bin/bash

log_file=$1

tmp=`cat $1 | grep "per iter" | awk 'BEGIN{cnt=0;T=0}{cnt+=1;T+=$18}END{print T/1000,cnt}'`

T=$(echo $tmp | cut -f1 -d" ")
cnt=$(echo $tmp | cut -f2 -d" ")

if [ $cnt != 200 ]; then
    # echo "broken log file" $1 $cnt
    echo "OOM"
    exit 0
fi

echo $T
