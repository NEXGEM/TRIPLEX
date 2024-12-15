# RAW_DIR=/path/to/raw/dir
# PROCESSED_DIR=/path/to/result/dir

RAW_DIR=/home/shared/image/public/TCGA/GBM 
PROCESSED_DIR=input/TCGA/GBM


# Preprocess ST data for inference (only WSI)
# python src/preprocess/CLAM/create_patches_fp.py \
#         --source $RAW_DIR \
#         --save_dir $PROCESSED_DIR \
#         --patch_size 256 \
#         --seg \
#         --patch \
#         --stitch \
#         --patch_level 0

EXTENSION='.svs'

python src/preprocess/prepare_data.py --input_dir $RAW_DIR \
                                --output_dir $PROCESSED_DIR \
                                --mode inference \
                                --patch_size 256 \
                                --slide_level 0 \
                                --slide_ext $EXTENSION



# ### Global features
# python src/preprocess/extract_img_features.py  \
#         --wsi_dataroot $RAW_DIR \
#         --patch_dataroot $PROCESSED_DIR'/patches' \
#         --embed_dataroot $PROCESSED_DIR'/emb/global' \
#         --slide_ext $EXTENSION \
#         --use_openslide \
#         --num_n 1

# ### Neighbor features
# python src/preprocess/extract_img_features.py  \
#         --wsi_dataroot $RAW_DIR \
#         --patch_dataroot $PROCESSED_DIR'/patches' \
#         --embed_dataroot $PROCESSED_DIR'/emb/neighbor' \
#         --slide_ext $EXTENSION \
#         --use_openslide \
#         --num_n 5
        