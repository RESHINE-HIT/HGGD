CUDA_VISIBLE_DEVICES=0 python test_graspnet.py \
--center-num 32 \
--anchor-num 7 \
--anchor-k 6 \
--anchor-w 50 \
--anchor-z 20 \
--grid-size 8 \
--scene-l 100 \
--scene-r 101 \
--all-points-num 25600 \
--group-num 512 \
--local-k 10 \
--ratio 8 \
--input-h 360 \
--input-w 640 \
--local-thres 0.01 \
--heatmap-thres 0.01 \
--num-workers 4 \
--dataset-path './dataset_ckpt/realsense/6dto2drefine_realsense' \
--checkpoint './logs/241206_170952_realsense/epoch_14_iou_0.984_cover_0.611' \
--scene-path './dataset_ckpt/graspnet' \
--dump-dir 'pred_grasps' \
--description 'realsense_seen' \
#--localnet 'PointMultiGraspNet_V3' \
