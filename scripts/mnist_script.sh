#!/bin/bash
set -e
#if [ -z "$1" ]
#then
#    echo "Please enter an optimizer's name"
#    exit 1
#fi
optimizer=fastrn
#p=1/3

for denom in 3 4 5 6 7
do
  python main.py --dataset mnist --model simple --optimizer $optimizer --storm_p 1/${denom} --lr 1e-2 --storm_a_0 -1 \
  --storm_per_coord --wandb
done

