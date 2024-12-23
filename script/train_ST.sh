MODEL=TRIPLEX
NUM_GPU=2
MODE=cv

datasets=( andersson andrew bryan )
for dataset in  "${datasets[@]}"; do
    echo Run training for 'ST/'$dataset'/TRIPLEX'
    sbatch slurm.sh 'ST/'$dataset'/'$MODEL $NUM_GPU $MODE
done
