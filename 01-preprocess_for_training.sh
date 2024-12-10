# RAW_DIR=/path/to/raw/dir
# PROCESSED_DIR=/path/to/result/dir

RAW_DIR=/home/shared/spRNAseq/visium/inhouse/SMC/NSCLC
PROCESSED_DIR=input/smc/lung

# Preprocess ST data for training

## Prepare patches and st data
# python preprocess/prepare_st_data.py --input_dir $RAW_DIR --output_dir $PROCESSED_DIR

## Prepare geneset for training
# python preprocess/get_geneset.py --st_dir $PROCESSED_DIR'/st' --output_dir $PROCESSED_DIR

## Extract features for TRIPLEX
### Global features
# python preprocess/extract_img_features.py  \
#         --patch_dataroot $PROCESSED_DIR'/patches' \
#         --embed_dataroot $PROCESSED_DIR'/emb/global' \
#         --id_path $PROCESSED_DIR'/ids.csv' \
#         --num_n 1

### Neighbor features
EXTENSION='.tif'
python preprocess/extract_img_features.py \
        --wsi_dataroot $RAW_DIR \
        --patch_dataroot $PROCESSED_DIR'/patches' \
        --embed_dataroot $PROCESSED_DIR'/emb/neighbor' \
        --id_path $PROCESSED_DIR'/ids.csv' \
        --slide_ext $EXTENSION \
        --num_n 5
