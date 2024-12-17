
# ['CCRCC', 'COAD', 'HCC', 'IDC', 'LUNG', 'LYMPG']
# datasets=$(ls config/hest)
# for dataset in $datasets; do
#     CUDA_VISIBLE_DEVICES=4,5 python src/main.py --config_name hest/$dataset/StNet --gpu 2 --mode cv
# done

# datasets=( IDC LYMPH_IDC )
# for dataset in "${datasets[@]}"; do
#     CUDA_VISIBLE_DEVICES=4,5 python src/main.py --config_name hest/$dataset/StNet --gpu 2 --mode cv
# done

datasets=$(ls config/hest)
for dataset in $datasets; do
    CUDA_VISIBLE_DEVICES=7 python src/main.py --config_name hest/$dataset/StNet --gpu 1 --mode eval
done