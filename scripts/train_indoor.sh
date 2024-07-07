DATA_ROOT=/home/MSDFNet/Dataset_processed
TRAIN_SET=$DATA_ROOT/VIVID_320
GPU_ID=0

CUDA_VISIBLE_DEVICES=${GPU_ID} \
python train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
--scene_type indoor \
-b 4 \
--sequence-length 3 \
--with-ssim 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--rearrange-bin 30 \
--clahe-clip 3.0 \
--log-output \
--name MSDFNet_indoor

