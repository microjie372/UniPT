

# pretraining
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/MDF/dist_train_mae_3db.sh 8 --cfg_file cfgs/MDF/uni_mae_voxel_rcnn_knw.yaml --batch_size 8 --workers 4 --epoch 3

# finetuning
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/MDF/dist_train_mdf_3db.sh 6 --cfg_file cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_uni3d.yaml --pretrained_model ../output/MDF/uni_mae_voxel_rcnn_knw/default/ckpt/checkpoint_epoch_3.pth --batch_size 12 --epoch 30 --workers 6