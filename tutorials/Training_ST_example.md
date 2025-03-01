# Training with ST data

This tutorial guides you through processing the ST dataset and training multiple models.

## Preprocess data

### Data preparation & Preprocessing for TRIPLEX
- BC1 dataset (Andersson et al.):
```bash
bash script/02.1-preprocess_BC1.sh
```
- BC2 dataset (Bryan et al.):
```bash
bash script/02.2-preprocess_BC2.sh
```
- SCC dataset (Andrew et al.):
```bash
bash script/02.3-preprocess_SCC.sh
```

### Preprocessing for EGN
```bash
python src/model/EGN/build_exemplar.py --data_dir input/ST/andersson
python src/model/EGN/build_exemplar.py --data_dir input/ST/bryan
python src/model/EGN/build_exemplar.py --data_dir input/ST/andrew
```

## Model Training

Define a dataset to be used for training
```bash
DATASET="ST/andersson"
```

Run the following script to train multiple models using cross-validation:

```bash
NUM_GPU=2
MODE=cv

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

# Loop through each model
for MODEL in "${!MODELS[@]}"; do
     LOG_NAME=${MODELS[$MODEL]}
     python src/main.py --config_name $DATASET/$MODEL --gpu 1 --mode eval --log_name $LOG_NAME
     python src/experiment/agg_results.py --dataset $DATASET --model $MODEL --log_name $LOG_NAME
done
```