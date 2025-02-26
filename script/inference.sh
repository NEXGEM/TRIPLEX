


MODEL=TRIPLEX
CKPT='weights/TRIPLEX/epoch=19-val_target=0.5444.ckpt'
python src/main.py --gpu 1 --config_name lunit/lung/$MODEL --mode inference --ckpt_path $CKPT