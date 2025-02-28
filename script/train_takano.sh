#!/bin/bash

NUM_GPU=4
MODE=cv
MODELS=("TRIPLEX" "StNet" "EGN")

for MODEL in "${MODELS[@]}"; do
    sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE
done
