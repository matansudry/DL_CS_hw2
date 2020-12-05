#!/bin/bash

echo "***  STARTING EXPERIMENTS ***"

for L in 3 6 9 12
do
        exp_name="exp2"
        srun -c 2 --gres=gpu:1  --pty python -m hw2.experiments run-exp -n ${exp_name} -M ycn --bs-train 512 --early-stopping 5 --lr 0.0001 -K $K -L $L -P $((L/2+1)) -H 100 --batches 500
done

echo "*** DONE ***"

