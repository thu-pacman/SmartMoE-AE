#!/bin/bash

log_file=$1

T=`cat $1 | grep "training takes" | awk '{print $10}' | xargs -I {} date +'%s' -d "01/01/1970 {} UTC"`

echo $T
