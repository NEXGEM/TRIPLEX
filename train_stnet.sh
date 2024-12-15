
# ['CCRCC', 'COAD', 'HCC', 'IDC', 'LUNG', 'LYMPG']
datasets=$(ls config/hest)
for dataset in $datasets; do
    CUDA_VISIBLE_DEVICES=7 python src/main.py --config_name hest/$dataset/StNet
done
