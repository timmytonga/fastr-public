#!/bin/bash
set -e
#if [ -z "$1" ]; then
#  echo "Please enter a lr"
#  exit 1
#fi

for lr in 1 0.1 0.01 0.001 0.0001
do
  python main.py --optimizer fastrn --lr $lr --storm_normalized --wandb
done