# RAW_DIR=/path/to/raw/dir
# PROCESSED_DIR=/path/to/result/dir

RAW_DIR=/home/shared/image/inhouse/lunit_ICI/images
PROCESSED_DIR=input/lunit/lung


# Preprocess ST data for inference (only WSI)
python preprocess/CLAM/create_patches_fp.py \
        --source $RAW_DIR \
        --save_dir $PROCESSED_DIR \
        --patch_size 256 \
        --seg \
        --patch \
        --stitch \
        --patch_level 1

EXTENSION='.mrxs'

# ### Global features
# python preprocess/extract_img_features.py  \
#         --wsi_dataroot $RAW_DIR \
#         --patch_dataroot $PROCESSED_DIR'/patches' \
#         --embed_dataroot $PROCESSED_DIR'/emb/global' \
#         --slide_ext $EXTENSION \
#         --num_n 1

# ### Neighbor features
# python preprocess/extract_img_features.py  \
#         --wsi_dataroot $RAW_DIR \
#         --patch_dataroot $PROCESSED_DIR'/patches' \
#         --embed_dataroot $PROCESSED_DIR'/emb/neighbor' \
#         --slide_ext $EXTENSION \
#         --use_openslide \
#         --num_n 5
        