
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,4,5,6,7
CUDA_VISIBLE_DEVICES=0 bash script/04-preprocess_for_inference.sh /home/shared/image/inhouse/lunit_ICI/images ./input/lunit/lung .mrxs 1 8 & 
CUDA_VISIBLE_DEVICES=1 bash script/04-preprocess_for_inference.sh /home/shared/image/inhouse/lunit_ICI/images ./input/lunit/lung .mrxs 1 8 &
CUDA_VISIBLE_DEVICES=2 bash script/04-preprocess_for_inference.sh /home/shared/image/inhouse/lunit_ICI/images ./input/lunit/lung .mrxs 1 8 &
CUDA_VISIBLE_DEVICES=3 bash script/04-preprocess_for_inference.sh /home/shared/image/inhouse/lunit_ICI/images ./input/lunit/lung .mrxs 1 8 &
CUDA_VISIBLE_DEVICES=4 bash script/04-preprocess_for_inference.sh /home/shared/image/inhouse/lunit_ICI/images ./input/lunit/lung .mrxs 1 8 & 
CUDA_VISIBLE_DEVICES=5 bash script/04-preprocess_for_inference.sh /home/shared/image/inhouse/lunit_ICI/images ./input/lunit/lung .mrxs 1 8 & 
CUDA_VISIBLE_DEVICES=6 bash script/04-preprocess_for_inference.sh /home/shared/image/inhouse/lunit_ICI/images ./input/lunit/lung .mrxs 1 8 & 
CUDA_VISIBLE_DEVICES=7 bash script/04-preprocess_for_inference.sh /home/shared/image/inhouse/lunit_ICI/images ./input/lunit/lung .mrxs 1 8 & 

