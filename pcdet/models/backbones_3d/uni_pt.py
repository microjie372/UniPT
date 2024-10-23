from functools import partial
# import random
# import numpy as np
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from ...utils import loss_utils
from .spconv_backbone import post_act_block
from ...utils import uni3d_norm_2_in
# from pcdet.utils import common_utils

def post_act_block_SP(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm'):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        # norm_fn(out_channels),
        # nn.ReLU(),
    )

    return m

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class PT3_VoxelBackBone8x(nn.Module):  
    """
    pre-trained model
    """
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.mask_ratio = model_cfg.MASKED_RATIO
        self.grid = model_cfg.GRID

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.source_one_name = model_cfg.SOURCE_ONE_NAME

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.num_point_features = 16

        self.decoder_s1 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, stride=1),           
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv_s1 = nn.Conv2d(256, 3 * 20, 1)
        self.num_conv_s1 = nn.Conv2d(256, 1, 1)

        self.decoder_s2 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv_s2 = nn.Conv2d(256, 3 * 20, 1)
        self.num_conv_s2 = nn.Conv2d(256, 1, 1)

        self.decoder_s3 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv_s3 = nn.Conv2d(256, 3 * 20, 1)
        self.num_conv_s3 = nn.Conv2d(256, 1, 1)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, 3, padding=1, output_padding=1, stride=(4, 2, 2), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, padding=(3,1,1), output_padding=1, stride=(3, 2, 2), bias=False),
        )

        down_factor = 8
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)

        voxel_size = model_cfg.VOXEL_SIZE
        point_cloud_range = model_cfg.POINT_CLOUD_RANGE
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = point_cloud_range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)
        self.criterion = nn.BCEWithLogitsLoss()

        self.mask_token = nn.Parameter(torch.zeros(1, 3))
        self.forward_re_dict = {}

    def get_loss_s1(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        pred_coor = self.forward_re_dict['pred_coor_s1']
        gt_coor = self.forward_re_dict['gt_coor_s1'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask_s1'].detach()

        pred_num = self.forward_re_dict['pred_num_s1']
        gt_num = self.forward_re_dict['gt_num_s1'].detach()

        gt_mask = self.forward_re_dict['gt_mask_s1'].detach()

        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)
        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss_occ = self.criterion(pred, target)

        loss = loss_num + loss_coor + loss_occ

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
            'loss_occ': loss_occ.item(),
        }

        return loss, tb_dict

    def get_loss_s2(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        pred_coor = self.forward_re_dict['pred_coor_s2']
        gt_coor = self.forward_re_dict['gt_coor_s2'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask_s2'].detach()

        pred_num = self.forward_re_dict['pred_num_s2']
        gt_num = self.forward_re_dict['gt_num_s2'].detach()

        gt_mask = self.forward_re_dict['gt_mask_s2'].detach()
    
        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)
        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss_occ = self.criterion(pred, target)

        loss = loss_num + loss_coor + loss_occ

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
            'loss_occ': loss_occ.item(),
        }

        return loss, tb_dict
    
    def get_loss_s3(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        pred_coor = self.forward_re_dict['pred_coor_s3']
        gt_coor = self.forward_re_dict['gt_coor_s3'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask_s3'].detach()

        pred_num = self.forward_re_dict['pred_num_s3']
        gt_num = self.forward_re_dict['gt_num_s3'].detach()

        gt_mask = self.forward_re_dict['gt_mask_s3'].detach()

        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)
        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss_occ = self.criterion(pred, target)

        loss = loss_num + loss_coor + loss_occ

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
            'loss_occ': loss_occ.item(),
        }

        return loss, tb_dict

    def get_num_loss(self, pred, target, mask):
        bs = pred.shape[0]
        loss = self.num_loss(pred, target).squeeze()
        if bs == 1:
            loss = loss.unsqueeze(dim=0)

        assert loss.size() == mask.size()
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def get_coor_loss(self, pred, target, mask, chamfer_mask):

        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)

        pred = pred.permute(0, 2, 3, 1)
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)

        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]

        pred = pred.reshape(-1, 3, 20).permute(0, 2, 1)
        target = target.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss

    def decode_feat_s1(self, feats, mask=None):
        if mask is not None:
            bs, c, h, w = feats.shape
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder_s1(feats)
        bs, c, h, w = x.shape
        coor = self.coor_conv_s1(x)
        num = self.num_conv_s1(x)
   
        return coor, num

    def decode_feat_s2(self, feats, mask=None):
        if mask is not None:
            bs, c, h, w = feats.shape
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder_s2(feats)
        bs, c, h, w = x.shape
        coor = self.coor_conv_s2(x)
        num = self.num_conv_s2(x)

        return coor, num

    def decode_feat_s3(self, feats, mask=None):
        if mask is not None:
            bs, c, h, w = feats.shape
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder_s3(feats)
        bs, c, h, w = x.shape
        coor = self.coor_conv_s3(x)
        num = self.num_conv_s3(x)

        return coor, num

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict[
            'voxel_num_points']

        coor_down_sample = coors.int().detach().clone()
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:] // (self.down_factor * self.grid)
        coor_down_sample[:, 1] = coor_down_sample[:, 1] // (coor_down_sample[:, 1].max() * 2)

        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0)

        select_ratio = 1 - self.mask_ratio          # ratio for select voxel
        nums = unique_coor_down_sample.shape[0]

        len_keep = int(nums * select_ratio)
        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)

        keep = ids_shuffle[:len_keep]
        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach()
        unique_keep_bool[keep] = 1

        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index)
        ids_keep = ids_keep.bool()

        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        ## mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask, :], coors[ids_mask, :]

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0], 1).to(voxel_features_mask.device).detach()
        pts_mask = spconv.SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()

        pts_mask = pts_mask.detach()
        pts_mask = self.unshuffle(pts_mask)
        bev_mask = pts_mask.squeeze().max(dim=1)[0]
        batch_dict['gt_mask'] = bev_mask

        ### gt num
        pts_gt_num = spconv.SparseConvTensor(
            num_points.view(-1, 1).detach(),
            coors.int(),
            self.sparse_shape,
            batch_size
        ).dense()
        bs, _, d, h, w = pts_gt_num.shape

        pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w))
        pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor ** 2
        pts_gt_num = pts_gt_num.detach()
        batch_dict['gt_num'] = pts_gt_num

        ###
        voxels_large, num_points_large, coors_large = batch_dict['voxels_bev'], batch_dict['voxel_num_points_bev'], \
                                                      batch_dict['voxel_coords_bev'],
        f_center = torch.zeros_like(voxels_large[:, :, :3])

        f_center[:, :, 0] = (voxels_large[:, :, 0] - (
                coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (
                coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz

        voxel_count = f_center.shape[1]
        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0)
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center)
        f_center *= mask_num

        sparse_shape = [1, self.sparse_shape[1] // self.down_factor, self.sparse_shape[2] // self.down_factor, ]

        chamfer_mask = spconv.SparseConvTensor(
            mask_num.squeeze().detach(),
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense()
        batch_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)

        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(),
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense()  #

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w)
        batch_dict['gt_coor'] = pts_gt_coor

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep, :], coors[ids_keep, :]
        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1)
        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        nums = voxel_features.shape[0]
        voxel_fratures_all_one = torch.ones(nums, 1).to(voxel_features.device)

        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=coors.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        x_up1 = self.deconv1(out.dense())
        x_up2 = self.deconv2(x_up1)
        x_up3 = self.deconv3(x_up2)
        self.forward_re_dict['pred'] = x_up3

        feats = out.dense()
        bs, c, d, h, w = feats.shape
        feats = feats.reshape(bs, -1, h, w)
        batch_dict['instance_feat'] = feats

        split_tag_s1, split_tag_s2_pre = common_utils.split_batch_dict('waymo', batch_dict)
        batch_s1, batch_s2_pre = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2_pre, batch_dict)

        split_tag_s2, split_tag_s3 = common_utils.split_batch_dict(self.source_one_name, batch_s2_pre)    #kitti
        batch_s2, batch_s3 = common_utils.split_two_batch_dict_gpu(split_tag_s2, split_tag_s3, batch_s2_pre)

        self.forward_re_dict['gt_mask_s1'] = batch_s1['gt_mask']
        self.forward_re_dict['gt_num_s1'] = batch_s1['gt_num']
        self.forward_re_dict['chamfer_mask_s1'] = batch_s1['chamfer_mask']
        self.forward_re_dict['gt_coor_s1'] = batch_s1['gt_coor']

        self.forward_re_dict['gt_mask_s2'] = batch_s2['gt_mask']
        self.forward_re_dict['gt_num_s2'] = batch_s2['gt_num']
        self.forward_re_dict['chamfer_mask_s2'] = batch_s2['chamfer_mask']
        self.forward_re_dict['gt_coor_s2'] = batch_s2['gt_coor']

        self.forward_re_dict['gt_mask_s3'] = batch_s3['gt_mask']
        self.forward_re_dict['gt_num_s3'] = batch_s3['gt_num']
        self.forward_re_dict['chamfer_mask_s3'] = batch_s3['chamfer_mask']
        self.forward_re_dict['gt_coor_s3'] = batch_s3['gt_coor']

        pred_coor_s1, pred_num_s1 = self.decode_feat_s1(batch_s1['instance_feat'])
        self.forward_re_dict['pred_coor_s1'] = pred_coor_s1
        self.forward_re_dict['pred_num_s1'] = pred_num_s1

        pred_coor_s2, pred_num_s2 = self.decode_feat_s2(batch_s2['instance_feat'])
        self.forward_re_dict['pred_coor_s2'] = pred_coor_s2
        self.forward_re_dict['pred_num_s2'] = pred_num_s2

        pred_coor_s3, pred_num_s3 = self.decode_feat_s3(batch_s3['instance_feat'])
        self.forward_re_dict['pred_coor_s3'] = pred_coor_s3
        self.forward_re_dict['pred_num_s3'] = pred_num_s3

        return batch_dict

class PT2_VoxelBackBone8x_UniBN(nn.Module): 
    """
    pre-trained model
    """
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(uni3d_norm_2_in.UniNorm1d, dataset_from_flag=int(self.model_cfg.db_source), eps=1e-3, momentum=0.01, voxel_coord=True)

        self.mask_ratio = model_cfg.MASKED_RATIO
        self.grid = model_cfg.GRID

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.source_one_name = model_cfg.SOURCE_ONE_NAME

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
        )
        self.bn_input = norm_fn(16)
        self.relu_input = nn.ReLU()

        block = post_act_block_SP

        # ----------Block_1---------#
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, padding=1, indice_key='subm1'),
        )
        self.conv1_bn_1 = norm_fn(16)
        self.conv1_relu_1 = nn.ReLU()

        # ----------Block_2---------#
        self.conv2_1 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        )
        self.conv2_bn_1 = norm_fn(32)
        self.conv2_relu_1 = nn.ReLU()

        self.conv2_2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, padding=1, indice_key='subm2'),
        )
        self.conv2_bn_2 = norm_fn(32)
        self.conv2_relu_2 = nn.ReLU()

        self.conv2_3 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 32, 3, padding=1, indice_key='subm2'),
        )
        self.conv2_bn_3 = norm_fn(32)
        self.conv2_relu_3 = nn.ReLU()

        # ----------Block_3---------#
        self.conv3_1 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        )
        self.conv3_bn_1 = norm_fn(64)
        self.conv3_relu_1 = nn.ReLU()

        self.conv3_2 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 64, 3, padding=1, indice_key='subm3'),
        )
        self.conv3_bn_2 = norm_fn(64)
        self.conv3_relu_2 = nn.ReLU()

        self.conv3_3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 64, 3, padding=1, indice_key='subm3'),
        )
        self.conv3_bn_3 = norm_fn(64)
        self.conv3_relu_3 = nn.ReLU()

        # ----------Block_4---------#
        self.conv4_1 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        )
        self.conv4_bn_1 = norm_fn(64)
        self.conv4_relu_1 = nn.ReLU()

        self.conv4_2 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, padding=1, indice_key='subm4'),
        )
        self.conv4_bn_2 = norm_fn(64)
        self.conv4_relu_2 = nn.ReLU()

        self.conv4_3 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, padding=1, indice_key='subm4'),
        )
        self.conv4_bn_3 = norm_fn(64)
        self.conv4_relu_3 = nn.ReLU()

        # ----------Last Block---------#
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
        )
        self.conv_out_bn = norm_fn(128)
        self.conv_out_relu = nn.ReLU()

        self.num_point_features = 16

        self.decoder_s1 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv_s1 = nn.Conv2d(256, 3 * 20, 1)
        self.num_conv_s1 = nn.Conv2d(256, 1, 1)

        self.decoder_s2 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv_s2 = nn.Conv2d(256, 3 * 20, 1)
        self.num_conv_s2 = nn.Conv2d(256, 1, 1)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, 3, padding=1, output_padding=1, stride=(4, 2, 2), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, padding=(3,1,1), output_padding=1, stride=(3, 2, 2), bias=False),
        )

        down_factor = 8
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)

        voxel_size = model_cfg.VOXEL_SIZE
        point_cloud_range = model_cfg.POINT_CLOUD_RANGE
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = point_cloud_range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)

        self.mask_token = nn.Parameter(torch.zeros(1, 3))

        self.criterion = nn.BCEWithLogitsLoss()
        self.forward_re_dict = {}

    def get_loss_s1(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        pred_coor = self.forward_re_dict['pred_coor_s1']
        gt_coor = self.forward_re_dict['gt_coor_s1'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask_s1'].detach()

        pred_num = self.forward_re_dict['pred_num_s1']
        gt_num = self.forward_re_dict['gt_num_s1'].detach()

        gt_mask = self.forward_re_dict['gt_mask_s1'].detach()

        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)
        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss_occ = self.criterion(pred, target)

        loss = loss_num + loss_coor + loss_occ

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
            'loss_occ': loss_occ.item(),
        }

        return loss, tb_dict

    def get_loss_s2(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        pred_coor = self.forward_re_dict['pred_coor_s2']
        gt_coor = self.forward_re_dict['gt_coor_s2'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask_s2'].detach()

        pred_num = self.forward_re_dict['pred_num_s2']
        gt_num = self.forward_re_dict['gt_num_s2'].detach()

        gt_mask = self.forward_re_dict['gt_mask_s2'].detach()

        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)
        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss_occ = self.criterion(pred, target)

        loss = loss_num + loss_coor + loss_occ

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
            'loss_occ': loss_occ.item(),
        }

        return loss, tb_dict

    def get_num_loss(self, pred, target, mask):
        bs = pred.shape[0]
        loss = self.num_loss(pred, target).squeeze()
        if bs == 1:
            loss = loss.unsqueeze(dim=0)

        assert loss.size() == mask.size()
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def get_coor_loss(self, pred, target, mask, chamfer_mask):

        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)

        pred = pred.permute(0, 2, 3, 1)
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)

        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]

        pred = pred.reshape(-1, 3, 20).permute(0, 2, 1)
        target = target.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss

    def decode_feat_s1(self, feats, mask=None):
        if mask is not None:
            bs, c, h, w = feats.shape
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder_s1(feats)
        bs, c, h, w = x.shape
        coor = self.coor_conv_s1(x)
        num = self.num_conv_s1(x)

        return coor, num

    def decode_feat_s2(self, feats, mask=None):
        if mask is not None:
            bs, c, h, w = feats.shape
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder_s2(feats)
        bs, c, h, w = x.shape
        coor = self.coor_conv_s2(x)
        num = self.num_conv_s2(x)

        return coor, num

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict[
            'voxel_num_points']

        coor_down_sample = coors.int().detach().clone()
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:] // (self.down_factor * self.grid)
        coor_down_sample[:, 1] = coor_down_sample[:, 1] // (coor_down_sample[:, 1].max() * 2)

        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0)

        select_ratio = 1 - self.mask_ratio  # ratio for select voxel
        nums = unique_coor_down_sample.shape[0]

        len_keep = int(nums * select_ratio)

        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)

        keep = ids_shuffle[:len_keep]

        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach()
        unique_keep_bool[keep] = 1

        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index)
        ids_keep = ids_keep.bool()
        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        ### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask, :], coors[ids_mask, :]

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0], 1).to(voxel_features_mask.device).detach()
        pts_mask = spconv.SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()

        pts_mask = pts_mask.detach()
        pts_mask = self.unshuffle(pts_mask)
        bev_mask = pts_mask.squeeze().max(dim=1)[0]

        batch_dict['gt_mask'] = bev_mask

        ## gt num
        pts_gt_num = spconv.SparseConvTensor(
            num_points.view(-1, 1).detach(),
            coors.int(),
            self.sparse_shape,
            batch_size
        ).dense()
        bs, _, d, h, w = pts_gt_num.shape

        pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w))
        pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor ** 2
        pts_gt_num = pts_gt_num.detach()
        batch_dict['gt_num'] = pts_gt_num

        ###
        voxels_large, num_points_large, coors_large = batch_dict['voxels_bev'], batch_dict['voxel_num_points_bev'], \
                                                      batch_dict['voxel_coords_bev'],
        f_center = torch.zeros_like(voxels_large[:, :, :3])
        f_center[:, :, 0] = (voxels_large[:, :, 0] - (
                coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (
                coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz

        voxel_count = f_center.shape[1]
        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0)
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center)
        f_center *= mask_num

        sparse_shape = [1, self.sparse_shape[1] // self.down_factor, self.sparse_shape[2] // self.down_factor, ]

        chamfer_mask = spconv.SparseConvTensor(
            mask_num.squeeze().detach(),
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense()
        batch_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)

        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(),
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense()  #

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w)
        batch_dict['gt_coor'] = pts_gt_coor
        ###

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep, :], coors[ids_keep, :]
        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1)

        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        nums = voxel_features.shape[0]
        voxel_fratures_all_one = torch.ones(nums, 1).to(voxel_features.device)

        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=coors.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # ----------Input Block---------#
        t_input = self.conv_input(input_sp_tensor)
        t_input = replace_feature(t_input, self.bn_input(t_input.features, t_input.indices))
        t_input = replace_feature(t_input, self.relu_input(t_input.features))
        # ----------Block_1---------#
        t_conv1 = self.conv1(t_input)
        t_conv1 = replace_feature(t_conv1, self.conv1_bn_1(t_conv1.features, t_conv1.indices))
        t_conv1 = replace_feature(t_conv1, self.conv1_relu_1(t_conv1.features))
        # ----------Block_2---------#
        t_conv2_1 = self.conv2_1(t_conv1)
        t_conv2_1 = replace_feature(t_conv2_1, self.conv2_bn_1(t_conv2_1.features, t_conv2_1.indices))
        t_conv2_1 = replace_feature(t_conv2_1, self.conv2_relu_1(t_conv2_1.features))

        t_conv2_2 = self.conv2_2(t_conv2_1)
        t_conv2_2 = replace_feature(t_conv2_2, self.conv2_bn_2(t_conv2_2.features, t_conv2_2.indices))
        t_conv2_2 = replace_feature(t_conv2_2, self.conv2_relu_2(t_conv2_2.features))

        t_conv2_3 = self.conv2_3(t_conv2_2)
        t_conv2_3 = replace_feature(t_conv2_3, self.conv2_bn_3(t_conv2_3.features, t_conv2_3.indices))
        t_conv2_3 = replace_feature(t_conv2_3, self.conv2_relu_3(t_conv2_3.features))
        # ----------Block_3---------#
        t_conv3_1 = self.conv3_1(t_conv2_3)
        t_conv3_1 = replace_feature(t_conv3_1, self.conv3_bn_1(t_conv3_1.features, t_conv3_1.indices))
        t_conv3_1 = replace_feature(t_conv3_1, self.conv3_relu_1(t_conv3_1.features))

        t_conv3_2 = self.conv3_2(t_conv3_1)
        t_conv3_2 = replace_feature(t_conv3_2, self.conv3_bn_2(t_conv3_2.features, t_conv3_2.indices))
        t_conv3_2 = replace_feature(t_conv3_2, self.conv3_relu_2(t_conv3_2.features))

        t_conv3_3 = self.conv3_3(t_conv3_2)
        t_conv3_3 = replace_feature(t_conv3_3, self.conv3_bn_3(t_conv3_3.features, t_conv3_3.indices))
        t_conv3_3 = replace_feature(t_conv3_3, self.conv3_relu_3(t_conv3_3.features))
        # ----------Block_4---------#
        t_conv4_1 = self.conv4_1(t_conv3_3)
        t_conv4_1 = replace_feature(t_conv4_1, self.conv4_bn_1(t_conv4_1.features, t_conv4_1.indices))
        t_conv4_1 = replace_feature(t_conv4_1, self.conv4_relu_1(t_conv4_1.features))

        t_conv4_2 = self.conv4_2(t_conv4_1)
        t_conv4_2 = replace_feature(t_conv4_2, self.conv4_bn_2(t_conv4_2.features, t_conv4_2.indices))
        t_conv4_2 = replace_feature(t_conv4_2, self.conv4_relu_2(t_conv4_2.features))

        t_conv4_3 = self.conv4_3(t_conv4_2)
        t_conv4_3 = replace_feature(t_conv4_3, self.conv4_bn_3(t_conv4_3.features, t_conv4_3.indices))
        t_conv4_3 = replace_feature(t_conv4_3, self.conv4_relu_3(t_conv4_3.features))

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(t_conv4_3)
        out = replace_feature(out, self.conv_out_bn(out.features, out.indices))
        out = replace_feature(out, self.conv_out_relu(out.features))

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': t_conv1,
                'x_conv2': t_conv2_3,
                'x_conv3': t_conv3_3,
                'x_conv4': t_conv4_3,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        x_up1 = self.deconv1(out.dense())
        x_up2 = self.deconv2(x_up1)
        x_up3 = self.deconv3(x_up2)
        self.forward_re_dict['pred'] = x_up3

        feats = out.dense()
        bs, c, d, h, w = feats.shape
        feats = feats.reshape(bs, -1, h, w)
        batch_dict['instance_feat'] = feats

        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)
        batch_s1, batch_s2 = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2, batch_dict)

        self.forward_re_dict['gt_mask_s1'] = batch_s1['gt_mask']
        self.forward_re_dict['gt_num_s1'] = batch_s1['gt_num']
        self.forward_re_dict['chamfer_mask_s1'] = batch_s1['chamfer_mask']
        self.forward_re_dict['gt_coor_s1'] = batch_s1['gt_coor']

        self.forward_re_dict['gt_mask_s2'] = batch_s2['gt_mask']
        self.forward_re_dict['gt_num_s2'] = batch_s2['gt_num']
        self.forward_re_dict['chamfer_mask_s2'] = batch_s2['chamfer_mask']
        self.forward_re_dict['gt_coor_s2'] = batch_s2['gt_coor']

        pred_coor_s1, pred_num_s1 = self.decode_feat_s1(batch_s1['instance_feat'])
        self.forward_re_dict['pred_coor_s1'] = pred_coor_s1
        self.forward_re_dict['pred_num_s1'] = pred_num_s1

        pred_coor_s2, pred_num_s2 = self.decode_feat_s2(batch_s2['instance_feat'])
        self.forward_re_dict['pred_coor_s2'] = pred_coor_s2
        self.forward_re_dict['pred_num_s2'] = pred_num_s2

        return batch_dict

class PT2_VoxelBackBone8x(nn.Module):   
    """
    pre-trained model
    """
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.mask_ratio = model_cfg.MASKED_RATIO
        self.grid = model_cfg.GRID

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.source_one_name = model_cfg.SOURCE_ONE_NAME

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.num_point_features = 16

        self.decoder_s1 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv_s1 = nn.Conv2d(256, 3 * 20, 1)
        self.num_conv_s1 = nn.Conv2d(256, 1, 1)

        self.decoder_s2 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv_s2 = nn.Conv2d(256, 3 * 20, 1)
        self.num_conv_s2 = nn.Conv2d(256, 1, 1)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, 3, padding=1, output_padding=1, stride=(4, 2, 2), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, 3, padding=(3,1,1), output_padding=1, stride=(3, 2, 2), bias=False),
        )

        down_factor = 8
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)

        voxel_size = model_cfg.VOXEL_SIZE
        point_cloud_range = model_cfg.POINT_CLOUD_RANGE
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = point_cloud_range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)

        self.mask_token = nn.Parameter(torch.zeros(1, 3))

        self.criterion = nn.BCEWithLogitsLoss()
        self.forward_re_dict = {}

    def get_loss_s1(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        pred_coor = self.forward_re_dict['pred_coor_s1']
        gt_coor = self.forward_re_dict['gt_coor_s1'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask_s1'].detach()

        pred_num = self.forward_re_dict['pred_num_s1']
        gt_num = self.forward_re_dict['gt_num_s1'].detach()

        gt_mask = self.forward_re_dict['gt_mask_s1'].detach()

        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)
        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss_occ = self.criterion(pred, target)

        loss = loss_num + loss_coor + loss_occ

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
            'loss_occ': loss_occ.item(),
        }

        return loss, tb_dict

    def get_loss_s2(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        pred_coor = self.forward_re_dict['pred_coor_s2']
        gt_coor = self.forward_re_dict['gt_coor_s2'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask_s2'].detach()

        pred_num = self.forward_re_dict['pred_num_s2']
        gt_num = self.forward_re_dict['gt_num_s2'].detach()

        gt_mask = self.forward_re_dict['gt_mask_s2'].detach()

        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)
        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss_occ = self.criterion(pred, target)

        loss = loss_num + loss_coor + loss_occ

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
            'loss_occ': loss_occ.item(),
        }

        return loss, tb_dict

    def get_num_loss(self, pred, target, mask):
        bs = pred.shape[0]
        loss = self.num_loss(pred, target).squeeze()
        if bs == 1:
            loss = loss.unsqueeze(dim=0)

        assert loss.size() == mask.size()
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def get_coor_loss(self, pred, target, mask, chamfer_mask):

        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)

        pred = pred.permute(0, 2, 3, 1)
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)

        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]

        pred = pred.reshape(-1, 3, 20).permute(0, 2, 1)
        target = target.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss

    def decode_feat_s1(self, feats, mask=None):
        if mask is not None:
            bs, c, h, w = feats.shape
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder_s1(feats)
        bs, c, h, w = x.shape
        coor = self.coor_conv_s1(x)
        num = self.num_conv_s1(x)

        return coor, num

    def decode_feat_s2(self, feats, mask=None):
        if mask is not None:
            bs, c, h, w = feats.shape
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder_s2(feats)
        bs, c, h, w = x.shape
        coor = self.coor_conv_s2(x)
        num = self.num_conv_s2(x)

        return coor, num

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict[
            'voxel_num_points']

        coor_down_sample = coors.int().detach().clone()
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:] // (self.down_factor * self.grid)
        coor_down_sample[:, 1] = coor_down_sample[:, 1] // (coor_down_sample[:, 1].max() * 2)

        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0)

        select_ratio = 1 - self.mask_ratio  # ratio for select voxel
        nums = unique_coor_down_sample.shape[0]

        len_keep = int(nums * select_ratio)

        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)

        keep = ids_shuffle[:len_keep]

        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach()
        unique_keep_bool[keep] = 1

        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index)
        ids_keep = ids_keep.bool()
        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        ### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask, :], coors[ids_mask, :]

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0], 1).to(voxel_features_mask.device).detach()
        pts_mask = spconv.SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()

        pts_mask = pts_mask.detach()
        pts_mask = self.unshuffle(pts_mask)
        bev_mask = pts_mask.squeeze().max(dim=1)[0]

        batch_dict['gt_mask'] = bev_mask

        ## gt num
        pts_gt_num = spconv.SparseConvTensor(
            num_points.view(-1, 1).detach(),
            coors.int(),
            self.sparse_shape,
            batch_size
        ).dense()
        bs, _, d, h, w = pts_gt_num.shape

        pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w))
        pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor ** 2
        pts_gt_num = pts_gt_num.detach()
        batch_dict['gt_num'] = pts_gt_num

        ###
        voxels_large, num_points_large, coors_large = batch_dict['voxels_bev'], batch_dict['voxel_num_points_bev'], \
                                                      batch_dict['voxel_coords_bev'],
        f_center = torch.zeros_like(voxels_large[:, :, :3])
        f_center[:, :, 0] = (voxels_large[:, :, 0] - (
                coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (
                coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz

        voxel_count = f_center.shape[1]
        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0)
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center)
        f_center *= mask_num

        sparse_shape = [1, self.sparse_shape[1] // self.down_factor, self.sparse_shape[2] // self.down_factor, ]

        chamfer_mask = spconv.SparseConvTensor(
            mask_num.squeeze().detach(),
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense()
        batch_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)

        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(),
            coors_large.int(),
            sparse_shape,
            batch_size
        ).dense()  #

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w)
        batch_dict['gt_coor'] = pts_gt_coor
        ###

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep, :], coors[ids_keep, :]
        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1)

        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        nums = voxel_features.shape[0]
        voxel_fratures_all_one = torch.ones(nums, 1).to(voxel_features.device)

        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=coors.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        x_up1 = self.deconv1(out.dense())
        x_up2 = self.deconv2(x_up1)
        x_up3 = self.deconv3(x_up2)
        self.forward_re_dict['pred'] = x_up3

        feats = out.dense()
        bs, c, d, h, w = feats.shape
        feats = feats.reshape(bs, -1, h, w)
        batch_dict['instance_feat'] = feats

        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)
        batch_s1, batch_s2 = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2, batch_dict)

        self.forward_re_dict['gt_mask_s1'] = batch_s1['gt_mask']
        self.forward_re_dict['gt_num_s1'] = batch_s1['gt_num']
        self.forward_re_dict['chamfer_mask_s1'] = batch_s1['chamfer_mask']
        self.forward_re_dict['gt_coor_s1'] = batch_s1['gt_coor']

        self.forward_re_dict['gt_mask_s2'] = batch_s2['gt_mask']
        self.forward_re_dict['gt_num_s2'] = batch_s2['gt_num']
        self.forward_re_dict['chamfer_mask_s2'] = batch_s2['chamfer_mask']
        self.forward_re_dict['gt_coor_s2'] = batch_s2['gt_coor']

        pred_coor_s1, pred_num_s1 = self.decode_feat_s1(batch_s1['instance_feat'])
        self.forward_re_dict['pred_coor_s1'] = pred_coor_s1
        self.forward_re_dict['pred_num_s1'] = pred_num_s1

        pred_coor_s2, pred_num_s2 = self.decode_feat_s2(batch_s2['instance_feat'])
        self.forward_re_dict['pred_coor_s2'] = pred_coor_s2
        self.forward_re_dict['pred_num_s2'] = pred_num_s2

        return batch_dict

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator
