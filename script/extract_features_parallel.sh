#!/bin/bash
# Script to extract features in parallel using multiple GPUs

# Usage
# bash script/extract_features_parallel.sh /path/to/images /path/to/output .ext [num_gpus] [feature_type]

RAW_DIR=$1
PROCESSED_DIR=$2
EXTENSION=$3
TOTAL_GPUS=${4:-8}
FEATURE_TYPE=${5:-both}  # Can be 'global', 'neighbor', or 'both'

# Check if there are enough available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $AVAILABLE_GPUS -lt $TOTAL_GPUS ]; then
    echo "Warning: Requested $TOTAL_GPUS GPUs but only $AVAILABLE_GPUS are available"
    TOTAL_GPUS=$AVAILABLE_GPUS
fi

echo "Extracting features using $TOTAL_GPUS GPUs"

# Function to extract features
extract_features() {
    local gpu_id=$1
    local feature_type=$2
    local model_name=${3:-"cigar"}
    local num_n=$4

    CUDA_VISIBLE_DEVICES=$gpu_id python src/preprocess/extract_img_features.py \
        --wsi_dataroot $RAW_DIR \
        --patch_dataroot $PROCESSED_DIR/patches \
        --embed_dataroot $PROCESSED_DIR/emb/$feature_type \
        --slide_ext $EXTENSION \
        --num_n $num_n \
        --model_name $model_name \
        --total_gpus $TOTAL_GPUS
}

# Kill background processes on script exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Export global features
if [ "$FEATURE_TYPE" = "global" ] || [ "$FEATURE_TYPE" = "both" ]; then
    echo "Extracting global features..."
    for ((i=0; i<$TOTAL_GPUS; i++)); do
        extract_features $i global "cigar" 1 &
    done
    wait
    echo "Global feature extraction complete!"
fi

# Export neighbor features
if [ "$FEATURE_TYPE" = "neighbor" ] || [ "$FEATURE_TYPE" = "both" ]; then
    echo "Extracting neighbor features..."
    for ((i=0; i<$TOTAL_GPUS; i++)); do
        extract_features $i neighbor "cigar" 5 &
    done
    wait
    echo "Neighbor feature extraction complete!"
fi

echo "All feature extraction complete!"
