
MODEL=TRIPLEX
python src/main.py --config_name takano/xenium/$MODEL --gpu 1 --mode eval --log_dir 2025-02-21-15-49
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE

# MODEL=StNet
# python src/main.py --config_name takano/xenium/$MODEL --gpu 1 --mode eval --log_dir 2025-02-20-20-06

# MODEL=EGN
# python src/main.py --config_name takano/xenium/$MODEL --gpu 1 --mode eval --log_dir 2025-02-20-22-32

# MODEL=StNet
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE

# MODEL=EGN
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE

# MODEL=BLEEP
# sbatch --gres=gpu:$NUM_GPU --ntasks-per-node=$NUM_GPU --mem-per-gpu=64G --cpus-per-gpu=6 script/slurm.sh "takano/xenium/$MODEL" $NUM_GPU $MODE


