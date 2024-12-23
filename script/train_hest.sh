
ulimit -n 100000
export NCCL_P2P_LEVEL=NVL
MODEL=TRIPLEX

datasets=$(ls config/hest)
for dataset in $datasets; do
    CUDA_VISIBLE_DEVICES=6,7 python src/main.py --config_name hest/$dataset/$MODEL --gpu 2 --mode cv
done

# datasets=( HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM )
# for dataset in "${datasets[@]}"; do
#     CUDA_VISIBLE_DEVICES=6,7 python src/main.py \
#                 --config_name hest/$dataset/TRIPLEX  \
#                 --mode cv \
#                 --gpu 2 \
#                 --mode cv
# done
