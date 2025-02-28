# Training with Visium data

This tutorial guides you through downloading and processing the GSE240429 Visium dataset and training multiple models.

## Download data from GEO dataset

1. Navigate to the GEO dataset page:
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429

2. Download and unzip all files (if necessary) *except* the following:
    - barcodes.tsv.gz
    - features.tsv.gz
    - matrix.mtx.gz

## Preprocess data

### Preprocessing for TRIPLEX
```bash
bash script/03-preprocess_new.sh ./GSE240429 input/GSE240429 tiff visium
```

### Preprocessing for EGN
```bash
python src/model/EGN/build_exemplar.py --data_dir input/GSE240429
```

## Model Training

Run the following script to train multiple models using cross-validation:

```bash
NUM_GPU=2
MODE=cv
DATASET="GSE240429"

# Define models to train
MODELS=("TRIPLEX" "StNet" "EGN" "BLEEP")

# Submit jobs for each model
for MODEL in "${MODELS[@]}"; do
     python src/main.py --config_name $DATASET/$MODEL --gpu $NUM_GPU --mode $MODE
done
```

## Model Evaluation

After training, evaluate each model with the following script:

```bash
declare -A MODELS=(
     ["TRIPLEX"]="Log name for TRIPLEX"
     ["StNet"]="Log name for StNet"
     ["EGN"]="Log name for EGN"
     ["BLEEP"]="Log name for BLEEP"
)

DATASET="GSE240429"
# Loop through each model
for MODEL in "${!MODELS[@]}"; do
     LOG_NAME=${MODELS[$MODEL]}
     python src/main.py --config_name $DATASET/$MODEL --gpu 1 --mode eval --log_name $LOG_NAME
     python src/experiment/agg_results.py --dataset $DATASET --model $MODEL --log_name $LOG_NAME
done
```
