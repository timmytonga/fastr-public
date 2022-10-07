#!/bin/bash
set -e
#if [ -z "$1" ]
#then
#    echo "Please enter an optimizer's name"
#    #    exit 1
#    #fi
#    optimizer=fastrn
#    #p=1/3

for denom in '1/3' '2/5' '1/4' '1/5' '1/6' '1/7'
do
    python main.py --optimizer fastrn --storm_p ${denom} --lr 0.001 --storm_a_0 -1  --storm_per_coordinate --wandb
done


