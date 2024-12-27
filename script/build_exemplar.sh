

for data_dir in input/hest/bench_data/*; do
    CUDA_VISIBLE_DEVICES=1 python src/model/EGN/build_exemplar.py --data_dir $data_dir
done