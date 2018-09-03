#!/usr/bin/env bash
set -eux
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    echo 'start training agent on task' $e
    python test_behavior_cloning.py $e --num_rollouts 10 #--render
done
