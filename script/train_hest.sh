
#!/bin/bash
NUM_GPU=1
MODE=cv

run_training() {
    local model=$1
    shift
    local datasets=("$@")
    local sbatch_opts=""

    # Use GPU option for all models except TRIPLEX
    # if [[ "$model" != "TRIPLEX" ]]; then
    #     sbatch_opts="--gres=gpu:$NUM_GPU"
    # fi
    sbatch_opts="--gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=62G --cpus-per-gpu=6"

    for dataset in "${datasets[@]}"; do
        echo "Run training for ${dataset}/${model}"
        sbatch ${sbatch_opts} script/slurm.sh "${dataset}/${model}" $NUM_GPU $MODE
    done
}

datasets=( 
    hest/bench_data/CCRCC 
    hest/bench_data/COAD 
    hest/bench_data/HCC 
    hest/bench_data/IDC 
    hest/bench_data/LUNG 
    hest/bench_data/LYMPH_IDC 
    hest/bench_data/PAAD 
    hest/bench_data/PRAD 
    hest/bench_data/READ 
    hest/bench_data/SKCM 
)

# datasets=( bench_data/COAD bench_data/LYMPH_IDC )
run_training "TRIPLEX" "${datasets[@]}"
