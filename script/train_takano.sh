NUM_GPU=4
MODE=cv

# MODEL=TRIPLEX
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE

# MODEL=StNet
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE

# MODEL=EGN
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE

MODEL=BLEEP
sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE


