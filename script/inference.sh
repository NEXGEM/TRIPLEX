


# MODEL=TRIPLEX
# CKPT='weights/TRIPLEX/epoch=25-val_target=0.5430.ckpt'
MODEL=$1
CKPT='weights/'$MODEL'/'$2
python src/main.py --gpu 1 --config_name lunit/lung/$MODEL --mode inference --ckpt_path $CKPT


# MODEL=StNet
# CKPT='weights/StNet/epoch=26-val_target=0.4768.ckpt'
# python src/main.py --gpu 1 --config_name lunit/lung/$MODEL --mode inference --ckpt_path $CKPT