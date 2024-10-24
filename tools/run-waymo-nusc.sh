# pretraining
bash ./scripts/MDF/dist_train_unipt.sh 8 --cfg_file cfgs/MDF/waymo_nusc/unipt_voxel_rcnn_waymo_nusc.yaml --source_one_name waymo

# finetuning
bash ./scripts/MDF/dist_train_mdf.sh 8 --cfg_file cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_feat_3_FT.yaml --pretrained_model {your_file_path} --source_one_name waymo 

