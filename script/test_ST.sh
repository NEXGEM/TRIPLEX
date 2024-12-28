MODEL=TRIPLEX
NUM_GPU=1
MODE=eval

datasets=( andersson andrew bryan )
for dataset in  "${datasets[@]}"; do
    echo Run training for 'ST/'$dataset'/TRIPLEX'
    sbatch script/slurm.sh 'ST/'$dataset'/'$MODEL $NUM_GPU $MODE
done
