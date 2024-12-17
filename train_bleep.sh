
# ['CCRCC', 'COAD', 'HCC', 'IDC', 'LUNG', 'LYMPG']
# datasets=$(ls config/hest)
# for dataset in $datasets; do
#     CUDA_VISIBLE_DEVICES=2,3 python src/main.py --config_name hest/$dataset/BLEEP --gpu 2
# done

# dataset='IDC'
# CUDA_VISIBLE_DEVICES=2,3 python src/main.py --config_name hest/$dataset/BLEEP --gpu 2 --mode cv

dataset='IDC'
CUDA_VISIBLE_DEVICES=4 python src/main.py --config_name hest/$dataset/BLEEP --gpu 1 --mode eval

# datasets=$(ls config/hest)
# for dataset in $datasets; do
#     CUDA_VISIBLE_DEVICES=3 python src/main.py --config_name hest/$dataset/BLEEP --gpu 1 --mode eval
# done