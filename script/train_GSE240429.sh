#!/bin/bash
# Script to train multiple models on GSE240429 dataset using Slurm

# Set common parameters
NUM_GPU=2
MODE=cv
DATASET="GSE240429"
MEM_PER_GPU="64G"
CPUS_PER_GPU=6

# Define models to train
MODELS=("TRIPLEX" "StNet" "EGN" "BLEEP")

# Submit jobs for each model
for MODEL in "${MODELS[@]}"; do
    sbatch --gres=gpu:$NUM_GPU \
           --ntasks-per-node=$NUM_GPU \
           --mem-per-gpu=$MEM_PER_GPU \
           --cpus-per-gpu=$CPUS_PER_GPU \
           script/slurm.sh "$DATASET/$MODEL" $NUM_GPU $MODE
done
