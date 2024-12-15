for data in input/hest/bench_data/*; do

    CUDA_VISIBLE_DEVICES=6 bash 01-preprocess_for_training.sh $data

done