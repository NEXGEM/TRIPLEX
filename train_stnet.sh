
ulimit -n 100000
export NCCL_P2P_LEVEL=NVL

# datasets=$(ls config/hest)
# for dataset in $datasets; do
#     CUDA_VISIBLE_DEVICES=4,5 python src/main.py --config_name hest/$dataset/StNet --gpu 2 --mode cv
# done

# datasets=( CCRCC COAD HCC LUNG PAAD PRAD READ SKCM )
# datasets=( IDC LYMPH_IDC )
# for dataset in "${datasets[@]}"; do
#     CUDA_VISIBLE_DEVICES=4,5 python src/main.py --config_name hest/$dataset/StNet --gpu 2 --mode cv
# done

datasets=$(ls config/hest)
for dataset in $datasets; do
    CUDA_VISIBLE_DEVICES=6 python src/main.py --config_name hest/$dataset/StNet --gpu 1 --mode eval
done