#!/bin/bash

log=$1
output=$2

cat $log | awk 'BEGIN{last=""} /updated/{print last} {last=$0}' | sed "s/iter/\niter /g" | grep "iter" | awk '{print $2}' > $output