#!/usr/bin/env bash

###############################################################################
#                                  User input                                 #
###############################################################################
# problems, separates by space
PROBLEMS="mnist_mlp cifar10_3c3d fmnist_2c2d cifar100_allcnnc"

# number of runs
NUM_RUNS=10
RUNS=$(seq 1 $NUM_RUNS)

###############################################################################
#                             Execute runs & plot                             #
###############################################################################
# run
for problem in $PROBLEMS;
do
    for run in $RUNS;
    do
        echo "Run $run memory benchmark on $problem"
        python expensive.py $problem $run
        python optimized.py $problem $run
        python baseline.py $problem $run
    done
    python plot.py $problem
done
