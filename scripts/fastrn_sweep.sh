#!/bin/bash
set -e
if [ -z "$1" ]; then
  echo "Please enter a lr"
  exit 1
fi

lr=$1

for a0 in 1e5 5e5 1e6 5e6 1e7
do
  python main.py --optimizer fastrn --lr $lr --storm_a_0 $a0 --wandb
done