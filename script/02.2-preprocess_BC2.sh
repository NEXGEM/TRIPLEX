
PROCESSED_DIR=input/ST/bryan
EXTENSION='.tif'

# Preprocess ST data for training

## Prepare patches and st data
python src/preprocess/prepare_data.py --input_dir $PROCESSED_DIR \
                                --output_dir $PROCESSED_DIR \
                                --mode hest \
                                --save_neighbors

## Prepare geneset for training
python src/preprocess/get_geneset.py \
                        --st_dir $PROCESSED_DIR'/adata' \
                        --output_dir $PROCESSED_DIR \
                        --n_top_hvg 50 \
                        --n_top_heg 1000

## Extract features for TRIPLEX
### Global features
python src/preprocess/extract_img_features.py  \
        --wsi_dataroot $PROCESSED_DIR'/wsis' \
        --patch_dataroot $PROCESSED_DIR'/patches' \
        --embed_dataroot $PROCESSED_DIR'/emb/global' \
        --slide_ext $EXTENSION \
        --num_n 1 \
        --model_name 'cigar'

### Neighbor features
python src/preprocess/extract_img_features.py \
        --wsi_dataroot $PROCESSED_DIR'/wsis' \
        --patch_dataroot $PROCESSED_DIR'/patches' \
        --embed_dataroot $PROCESSED_DIR'/emb/neighbor' \
        --slide_ext $EXTENSION \
        --num_n 5 \
        --model_name 'cigar'
