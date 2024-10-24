import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader_mdf, build_dataloader, DatasetTemplate
from pcdet.models import build_network_multi_db, model_fn_decorator, load_data_to_gpu
from pcdet.utils import common_utils, calibration_kitti, object3d_kitti, box_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_multi_db_utils import train_model
from pcdet.datasets.waymo.waymo_utils import generate_labels

class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.infos = []
        self.data_path = self.root_path.parent

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            print(self.sample_file_list[index])
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        sequence_name = self.data_path.stem
        index = int(self.root_path.stem)
        waymo_infos = []
        print('frame_id is:', index)

        info_path = os.path.join(self.data_path, ('%s.pkl' % sequence_name))
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            waymo_infos.extend(infos)
        self.infos.extend(waymo_infos[:])

        info = copy.deepcopy(self.infos[index])

        input_dict = {
            'points': points,
            'frame_id': index,
            'db_flag': "waymo",
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']
            lidar_z = gt_boxes_lidar[:, 2]
            mask = (annos['num_points_in_gt'] > 5)  # filter empty boxes
            annos['name'] = annos['name'][mask]
            gt_boxes_lidar = gt_boxes_lidar[mask]
            annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            #print(gt_boxes_lidar.shape)

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        #print(data_dict['gt_boxes'].shape)
        return data_dict

class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, idx=0, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.index = idx
        self.infos = []
        self.data_path = self.root_path.parent.parent.parent

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        sequence_name = 'nuscenes_infos_10sweeps_val'
        index = self.root_path.stem
        #index = os.path.basename(self.root_path).split('.')[0]
        nuscenes_infos = []
        print('frame_id is:', index)

        info_path = os.path.join(self.data_path, ('%s.pkl' % sequence_name))

        input_dict = {
            'points': points,
            'frame_id': index,
            'db_flag': "nusc",
        }

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            nuscenes_infos.extend(infos)
        self.infos.extend(nuscenes_infos[:])

        print(self.index)
        info = copy.deepcopy(self.infos[self.index])
        print('The index of visualization is:', info['lidar_path'])

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                #mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                mask = (info['num_lidar_pts'] > 10)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
            
            #if self.dataset_cfg.get('SHIFT_COOR', None):
            #    input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
            
            # for debug only
            # gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            # debug_dict = {'gt_boxes': copy.deepcopy(input_dict['gt_boxes'][gt_boxes_mask])}
        
        data_dict = self.prepare_data(data_dict=input_dict)

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
            
        return data_dict

class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, idx=0, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.infos = []
        self.data_path = self.root_path.parent.parent.parent
        self.root_split_path = self.root_path.parent.parent
        self.index = idx

    def __len__(self):
        return len(self.sample_file_list)

    def get_lidar(self, idx):
        if self.oss_path is None:
            lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
            assert lidar_file.exists()
            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        else:
            lidar_file = os.path.join(self.root_split_path, 'velodyne', ('%s.bin' % idx))
            sdk_local_bytes = self.client.get(lidar_file, update_cache=True)
            points = np.frombuffer(sdk_local_bytes, dtype=np.float32).reshape(-1, 4).copy()

        return points

    def get_calib(self, idx):
        if self.oss_path is None:
            calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
            print(calib_file)
            assert calib_file.exists()
            calibrated_res = calibration_kitti.Calibration(calib_file, False)
        else:
            calib_file = os.path.join(self.root_split_path, 'calib', ('%s.txt' % idx))
            text_bytes = self.client.get(calib_file, update_cache=True)
            text_bytes = text_bytes.decode('utf-8')
            calibrated_res = calibration_kitti.Calibration(io.StringIO(text_bytes), True)
        return calibrated_res

    def get_fov_flag(pts_rect, img_shape, calib, margin=0):
        """
        Args:
            pts_rect:
            img_shape:
            calib:
            margin:
        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        sequence_name = 'kitti_infos_val'
        idx = os.path.basename(self.root_path).split('.')[0]
        kitti_infos = []
        #print('frame_id is:', self.index)

        info_path = os.path.join(self.data_path, ('%s.pkl' % sequence_name))
        print(info_path)

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            kitti_infos.extend(infos)
        self.infos.extend(kitti_infos[:])

        info = copy.deepcopy(self.infos[self.index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        print('frame_id is:', sample_idx)
        
        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'db_flag': "kitti",
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            
            #if self.dataset_cfg.get('SHIFT_COOR', None):
            #    gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            
        if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                #fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
                margin = 0
                val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
                val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
                val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
                pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
                points = points[pts_valid_flag]
        input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--source_one_name', type=str, default="nusc", help='enter the name of the first dataset of merged datasets')
    parser.add_argument('--source_1', type=int, default=2, help='if test the source_1 data')
    parser.add_argument('--index', type=int, default=0, help='specify the index of nusc val')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of Object Detection-------------------------')
    demo_dataset = WaymoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    demo_dataset_2 = NuScenesDataset(
       dataset_cfg=cfg.DATA_CONFIG_SRC_2, class_names=cfg.DATA_CONFIG_SRC_2.CLASS_NAMES, training=False,
       root_path=Path(args.data_path), ext=args.ext, logger=logger, idx=args.index
    )

    # demo_dataset_2 = KittiDataset(
    #     dataset_cfg=cfg.DATA_CONFIG_SRC_2, class_names=cfg.DATA_CONFIG_SRC_2.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), ext=args.ext, logger=logger, idx=args.index
    # )
    
    # add the dataset_source flag into Dual_BN layer
    if cfg.MODEL.get('POINT_T', None):
        cfg.MODEL.POINT_T.update({"db_source": args.source_1})
    if cfg.MODEL.get('BACKBONE_3D', None):
        cfg.MODEL.BACKBONE_3D.update({"db_source": args.source_1})
    if cfg.MODEL.get('DENSE_3D_MoE', None):
        cfg.MODEL.DENSE_3D_MoE.update({"db_source": args.source_1})
    if cfg.MODEL.get('BACKBONE_2D', None):
        cfg.MODEL.BACKBONE_2D.update({"db_source": args.source_1})
    if cfg.MODEL.get('DENSE_2D_MoE', None):
        cfg.MODEL.DENSE_2D_MoE.update({"db_source": args.source_1})
    if cfg.MODEL.get('PFE', None):
        cfg.MODEL.PFE.update({"db_source": args.source_1})
    
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network_multi_db(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), num_class_s2=len(cfg.DATA_CONFIG_SRC_2.CLASS_NAMES), \
         dataset=demo_dataset, dataset_s2=demo_dataset_2, source_one_name=args.source_one_name)
   
    if args.source_1 == 1:
        logger.info('**********************Testing Dataset=waymo**********************')
        
    elif args.source_1 == 2:
        logger.info('**********************Testing Dataset=nusc**********************')  #kitti
      
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])

            load_data_to_gpu(data_dict)
            pred_dicts, _, data_dict = model.forward(data_dict)
	       
            a = data_dict['points'][:, 1:].cpu()
            b = pred_dicts[0]['pred_boxes'].cpu()
            
            print('The number of predicted boxes is:', b.shape)
            
            c = pred_dicts[0]['pred_scores'].cpu()
            d = pred_dicts[0]['pred_labels'].cpu()
            e = data_dict['gt_boxes'].squeeze(0).cpu()
            
            print('The number of gt boxes is:', e.shape)

            np.save('./vis-det/points_demo.npy',a)
            np.save('./vis-det/ref_boxes.npy',b)
            np.save('./vis-det/ref_scores.npy',c)
            np.save('./vis-det/ref_labels.npy',d)
            np.save('./vis-det/gt_boxes.npy',e)


            #V.draw_scenes(
            #    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            #)

            #if not OPEN3D_FLAG:
            #    mlab.show(stop=True)

    logger.info('Demo done.')

if __name__ == '__main__':
    main()
