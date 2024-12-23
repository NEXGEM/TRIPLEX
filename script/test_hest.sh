
ulimit -n 100000
export NCCL_P2P_LEVEL=NVL

datasets=$(ls config/hest)
for dataset in $datasets; do
    CUDA_VISIBLE_DEVICES=7 python src/main.py --config_name hest/$dataset/TRIPLEX --gpu 1 --mode eval
done
