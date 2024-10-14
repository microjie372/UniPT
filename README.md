# UniPT
This is an official implementation of UniPT, full codes will be released soon
## Getting Started

### Installation & Requirements
All the codes are tested in the following environment:
* Python = 3.7.16
* CUDA = 11.1
* torch = 1.8.1+cu111
* torch-scatter= 2.0.6
* torchvision = 0.9.1+cu111
* spconv-cu111 = 2.1.25

a. Clone this repository.
```shell
git clone https://github.com/microjie372/UniPT.git
```

b. Install the dependent libraries as follows:

* Install the python dependent libraries.
  ```shell
    pip install -r requirements.txt 
  ```

* Install the gcc library, we use the gcc-5.4 version

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * It is recommended that you should install the latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).
    * Also, you should choice **the right version of spconv**, according to **your CUDA version**. For example, for CUDA 11.1, pip install spconv-cu111
  
c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

## Dataset Preparation

We provide the dataloader for Waymo, KITTI and NuScenes datasets.  

### Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
UniPT
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
├── pcdet
├── tools
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
# tf 2.0.0
pip3 install waymo-open-dataset-tf-2-0-0
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed): 
```python 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo/MDF/waymo_dataset.yaml
```

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

```
UniPT
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti/MDF/kitti_dataset.yaml
```

### NuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and 
organize the downloaded files as follows: 
```
UniPT
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```

* Generate the data infos by running the following command (it may take several hours): 
```python 
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes/MDF/nuscenes_dataset.yaml \
    --version v1.0-trainval
```

## Training-Testing for UniPT on Multi-dataset Object Detection

Here, we take Waymo-and-nuScenes consolidation as an example.

## Pre-training stage: train a UniPT model on the merged dataset: 

```shell script
bash scripts/MDF/dist_train_unipt.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/unipt_voxel_rcnn_waymo_nusc.yaml \
--source_one_name waymo
```

* Pre-train other networks such as PV-RCNN using multiple GPUs
```shell script
bash scripts/MDF/dist_train_unipt.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/unipt_pv_rcnn_waymo_nusc.yaml \
--source_one_name waymo
```

## Fine-tuning stage: train baseline detection models using above pre-trained networks on the merged dataset: 

```shell script
bash scripts/MDF/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_voxel_rcnn_feat_3_FT.yaml \
--pretrained_model {The path of your pre-trained checkpoint} \
--source_one_name waymo
```

* Fine-tune other detection models such as PV-RCNN using multiple GPUs
```shell script
bash scripts/MDF/dist_train_mdf.sh ${NUM_GPUs} \
--cfg_file ./cfgs/MDF/waymo_nusc/waymo_nusc_pv_rcnn_feat_3_FT.yaml \
--pretrained_model {The path of your pre-trained checkpoint} \
--source_one_name waymo
```

&ensp;
## Evaluation stage: evaluate the detection model on different datasets:
* Note that for the KITTI-related evaluation, please try --set DATA_CONFIG.FOV_POINTS_ONLY True to enable front view point cloud only.

- ${FIRST_DB_NAME} denotes that the fisrt dataset name of the merged two dataset, which is used to split the merged dataset into two individual datasets.

- ${DB_SOURCE} denotes the dataset to be tested.

* Test the models using multiple GPUs
```shell script
bash scripts/MDF/dist_test_mdf.sh ${NUM_GPUs} --cfg_file ${CFG_FILE} --ckpt ${CKPT} \
--source_one_name ${FIRST_DB_NAME} --source_1 ${DB_SOURCE} 
```
