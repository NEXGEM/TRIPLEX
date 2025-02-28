#!/bin/bash
NUM_GPU=1
MODE=cv

run_training() {
    local model=$1
    shift
    local datasets=("$@")
    local sbatch_opts=""

    # Use GPU option for all models except TRIPLEX
    if [[ "$model" != "TRIPLEX" ]]; then
        sbatch_opts="--gres=gpu:$NUM_GPU"
    fi

    for dataset in "${datasets[@]}"; do
        echo "Run training for ST/${dataset}/${model}"
        sbatch ${sbatch_opts} script/slurm.sh "ST/${dataset}/${model}" $NUM_GPU $MODE
    done
}

# TRIPLEX (no --gres required)
datasets=( andersson andrew bryan )
run_training "TRIPLEX" "${datasets[@]}"

# BLEEP
datasets=( andersson andrew bryan )
run_training "BLEEP" "${datasets[@]}"

# StNet
datasets=( andersson andrew bryan )
run_training "StNet" "${datasets[@]}"

# EGN (only andersson and andrew)
datasets=( andersson andrew )
run_training "EGN" "${datasets[@]}"
