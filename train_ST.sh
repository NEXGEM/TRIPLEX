

datasets=( andersson andrew bryan )
for dataset in  "${datasets[@]}"; do
    echo Run training for 'ST/'$dataset'/TRIPLEX'
    sbatch slurm.sh 'ST/'$dataset'/TRIPLEX'
done

# datasets=( bryan )
# for dataset in  "${datasets[@]}"; do
#     echo Run training for 'ST/'$dataset'/TRIPLEX'
#     sbatch slurm.sh 'ST/'$dataset'/TRIPLEX'
# done