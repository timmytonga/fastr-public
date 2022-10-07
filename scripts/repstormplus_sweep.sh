#!/bin/bash
set -e
if [ -z "$1" ]
then
    echo "Please enter a value for u"
    exit 1
fi

u=$1
for c in 10 1e2 1e3 1e4
do
  python main.py --optimizer repstormplus --lr 0.1 --storm_u $u --storm_c $c --wandb
done