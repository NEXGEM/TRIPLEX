NUM_GPU=1
MODE=cv

# MODEL=TRIPLEX
# datasets=( andersson andrew bryan )
# for dataset in  "${datasets[@]}"; do
#     echo Run training for 'ST/'$dataset'/'$MODEL
#     sbatch script/slurm.sh 'ST/'$dataset'/'$MODEL $NUM_GPU $MODE
# done

MODEL=BLEEP
datasets=( andersson andrew bryan )
for dataset in  "${datasets[@]}"; do
    echo Run training for 'ST/'$dataset'/'$MODEL
    sbatch --gres=gpu:$NUM_GPU script/slurm.sh 'ST/'$dataset'/'$MODEL $NUM_GPU $MODE
done

MODEL=StNet
datasets=( andersson andrew bryan )
for dataset in  "${datasets[@]}"; do
    echo Run training for 'ST/'$dataset'/'$MODEL
    sbatch --gres=gpu:$NUM_GPU script/slurm.sh 'ST/'$dataset'/'$MODEL $NUM_GPU $MODE
done

MODEL=EGN
datasets=( andersson andrew bryan )
for dataset in  "${datasets[@]}"; do
    echo Run training for 'ST/'$dataset'/'$MODEL
    sbatch --gres=gpu:$NUM_GPU script/slurm.sh 'ST/'$dataset'/'$MODEL $NUM_GPU $MODE
done
