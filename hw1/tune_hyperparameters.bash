#!/usr/bin/env bash
set -eux
touch result/bc_tune_param.txt
e='Hopper-v2'
for $lr in 1e
do
    echo 'start training agent with learning_rate' $lr 'steps' >> result/bc_tune_param.txt
    python behavior_cloning.py $e --learning_rate $lr
    python test_behavior_cloning.py $e --num_rollouts 10 >> result/bc_tune_param.txt
done