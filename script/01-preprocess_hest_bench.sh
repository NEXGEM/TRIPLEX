
RAW_DIR=$1
PROCESSED_DIR=$2
EXTENSION=$3

# Preprocess ST data for training

## Prepare patches and st data
python src/preprocess/prepare_data.py --input_dir $PROCESSED_DIR'/adata' \
                                --output_dir $PROCESSED_DIR \
                                --mode hest

## Extract features for TRIPLEX
### Global features
python src/preprocess/extract_img_features.py  \
        --patch_dataroot $PROCESSED_DIR'/patches' \
        --embed_dataroot $PROCESSED_DIR'/emb/global' \
        --num_n 1 

### Neighbor features
python src/preprocess/extract_img_features.py \
        --wsi_dataroot $RAW_DIR \
        --patch_dataroot $PROCESSED_DIR'/patches' \
        --embed_dataroot $PROCESSED_DIR'/emb/neighbor' \
        --slide_ext $EXTENSION \
        --num_n 5 
