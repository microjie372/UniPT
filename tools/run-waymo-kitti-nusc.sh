# pretraining
bash ./scripts/MDF/dist_train_unipt_3db.sh 8 --cfg_file cfgs/MDF/KNW/unipt_voxel_rcnn_knw.yaml

# finetuning
bash ./scripts/MDF/dist_train_mdf_3db.sh 8 --cfg_file cfgs/MDF/KNW/knw_voxel_rcnn_feat_3_FT.yaml --pretrained_model {your_file_path}
