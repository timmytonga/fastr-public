#!/bin/bash
set -e
if [ -z "$1" ]
then
    echo "Please enter an optimizer's name"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Setting default --storm_a_0 to adaptive"
    a0=-1
else
    a0=$2
fi


for lr in 1e-5 1e-4 1e-3 1e-2 1e-1 1 10
do
  # storm_a_0 = -1        : means adaptive a_0: it is set to largest stochastic grad norm
  # storm_ema_g           : set exponential moving average for sum_g in fastrn/storm+ and for sum_diff_g for fastrd
  #   note    -- for ema_g ONLY, it seems like we need to set a larger lr like 1, as oppose to the per coordinate update
  # storm_per_coordinate  : update per coordinate rather than norm
  #   note    -- for per_coordinate, it seems like we need to set a smaller lr like 1e-3.
  #   note 2  -- if set BOTH ema_g and per_coordinate, we might need to set to a smaller lr like 1e-3.
  python main.py --optimizer $optimizer --lr $lr --storm_a_0 $a0 --storm_ema_g --storm_per_coordinate --wandb
done

