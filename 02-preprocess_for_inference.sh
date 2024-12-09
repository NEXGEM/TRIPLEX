
# Preprocess ST data for inference (only WSI)
python preprocess/CLAM/create_patches_fp.py \
        --source $DATA_DIRECTORY \
        --save_dir $RESULTS_DIRECTORY \
        --patch_size 256 \
        --seg \
        --patch \
        --stitch \
        --patch_level 1

### Global features
python preprocess/extract_img_features.py  \
        --wsi_dataroot $INPUT_DIR'/patches' \
        --patch_dataroot $DIR_TO_COORDS'/patches' \
        --embed_dataroot $NEIGHBOR_FEATURES_DIRECTORY \
        --slide_ext $EXTENSION \
        --num_n 1

### Neighbor features
python preprocess/extract_img_features.py  \
        --wsi_dataroot $INPUT_DIR'/patches' \
        --patch_dataroot $DIR_TO_COORDS'/patches' \
        --embed_dataroot $NEIGHBOR_FEATURES_DIRECTORY \
        --slide_ext $EXTENSION \
        --use_openslide \
        --num_n 5
        