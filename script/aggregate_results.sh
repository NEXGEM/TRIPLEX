

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
    python src/experiment/agg_results.py --dataset $DATASET --model $MODEL --log_name $LOG_NAME
done



