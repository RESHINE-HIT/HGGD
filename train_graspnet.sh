CUDA_VISIBLE_DEVICES=0 python train_graspnet.py \
--batch-size 2 \
--step-cnt 2 \
--lr 1e-2 \
--anchor-num 7 \
--anchor-k 6 \
--anchor-w 50 \
--anchor-z 20 \
--all-points-num 25600 \
--group-num 512 \
--center-num 32 \
--scene-l 0 \
--scene-r 29 \
--noise 0 \
--grid-size 8 \
--input-w 640 \
--input-h 360 \
--loc-a 1 \
--reg-b 5 \
--cls-c 1 \
--offset-d 1 \
--epochs 15 \
--ratio 8 \
--num-workers 4 \
--save-freq 1 \
--optim 'adamw' \
--dataset-path './dataset_ckpt/realsense/6dto2drefine_realsense' \
--scene-path './dataset_ckpt/graspnet' \
--description 'realsense' \
--joint-trainning \
#--localnet 'PointMultiGraspNet_V3' \

