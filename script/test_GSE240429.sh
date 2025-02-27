
# Define models and their corresponding log directories
declare -A MODELS=(
    ["TRIPLEX"]="2025-02-26-16-28"
    ["StNet"]="2025-02-26-16-28"
    ["EGN"]="2025-02-26-16-32"
    ["BLEEP"]="2025-02-26-16-32"
)

DATASET="GSE240429"
# Loop through each model
for MODEL in "${!MODELS[@]}"; do
    LOG_NAME=${MODELS[$MODEL]}
    echo "MODEL: $MODEL"
    echo "LOG_NAME: $LOG_NAME"
    python src/main.py --config_name $DATASET/$MODEL --gpu 1 --mode eval --log_name $LOG_NAME
    python src/experiment/agg_results.py --dataset $DATASET --model $MODEL --log_name $LOG_NAME
done



# MODEL=StNet
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE

# MODEL=EGN
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE

# MODEL=BLEEP
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE


