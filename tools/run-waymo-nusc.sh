

# pretraining
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 bash ./scripts/MDF/dist_train_mae.sh 6 --cfg_file cfgs/MDF/uni_mae_waymo_nusc.yaml --source_one_name waymo --batch_size 6 --workers 2 --epoch 3

# finetuning
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 bash ./scripts/MDF/dist_train_mdf.sh 6 --cfg_file cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_feat_3_domain_attention.yaml --pretrained_model ../output/MDF/uni_mae_waymo_nusc/default/ckpt/checkpoint_epoch_3.pth --batch_size 18 --source_one_name waymo --epoch 30 --workers 2 

