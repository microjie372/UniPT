import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.utils import common_utils
#from pcdet.models.model_utils.weight_init import *

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Args:
        query: (batch_size, num_heads, seq_len_q, d_k), given sequence that we focus on
        key: (batch_size, num_heads, seq_len_k, d_k), the sequence to check relevance with query
        value: (batch_size, num_heads, seq_len_v, d_k), seq_len_k == seq_len_v, usually value and key come from the same source
        mask: for encoder, mask is [batch_size, 1, 1, seq_len_k], for decoder, mask is [batch_size, 1, seq_len_q, seq_len_k]
        dropout: nn.Dropout(), optional
    Returns:
        output: (batch_size, num_heads, seq_len_q, d_v), attn: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    # size of scores: (batch_size, num_heads, seq_len_q, seq_len_k)
    # scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = scores.softmax(dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return torch.matmul(scores, value)

class CrossBlock(nn.Module):
    def __init__(self, query_dim=512, key_dim=512, proj_dim=64, groups=1):
        super(CrossBlock, self).__init__()

        self.query = nn.Conv2d(in_channels=query_dim, out_channels=proj_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels=key_dim, out_channels=proj_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels=key_dim, out_channels=proj_dim, kernel_size=1)
        self.drop = nn.Dropout(p=0.0)

    def forward(self, query_x, ref_x):
        """
        Args:
            query_x (torch.Tensor): the query feature map
            ref_x (torch.Tensor): the reference feature map
        """
        batch_size, C, H, W = query_x.size()

        proj_query = self.query(query_x).view(batch_size, -1, H*W).unsqueeze(1)
        proj_key = self.key(ref_x).view(batch_size, -1 , H*W).unsqueeze(1)
        proj_value = self.value(ref_x).view(batch_size, -1, H*W).unsqueeze(1)

        out = scaled_dot_product_attention(proj_query, proj_key, proj_value, dropout=self.drop)
        out = out.squeeze(1).view(batch_size, -1, H, W)

        return out

class DENSE_2D_TRIPLE_ATT(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.source_one_name = self.model_cfg.SOURCE_ONE_NAME
        self.per_task_channels = self.model_cfg.INPUT_CONV_CHANNEL

        self.query_dim = self.model_cfg.INPUT_CONV_CHANNEL
        self.key_dim, self.proj_dim = self.query_dim, self.query_dim // 8

        self.se_1 = SEBlock(self.per_task_channels)
        self.se_2 = SEBlock(self.per_task_channels)

        self.cross_att = CrossBlock(self.query_dim, self.key_dim, self.proj_dim)

        self.db_source = int(self.model_cfg.db_source)

        self.dimentionality_upsample = nn.Sequential(
            nn.Conv2d(self.proj_dim, self.query_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.query_dim // 2, self.query_dim, kernel_size=1)
        )

    def forward(self, data_dict):
        # Get shared representation
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, data_dict)
        spatial_features_2d = data_dict['spatial_features_2d']

        spatial_features_2d_s1 = spatial_features_2d[split_tag_s1,:,:,:]
        spatial_features_2d_s2 = spatial_features_2d[split_tag_s2,:,:,:]

        if self.training:
            batch_size, C, H, W = spatial_features_2d_s1.size()

            out1 = self.cross_att(spatial_features_2d_s1, spatial_features_2d_s2)

            out_s1 = self.se_1(self.dimentionality_upsample(out1) + spatial_features_2d_s1)
            out_s2 = self.se_2(self.dimentionality_upsample(out1) + spatial_features_2d_s2)

            concat_f = torch.cat([out_s1, out_s2], 0)
            data_dict['spatial_features_2d'] = concat_f
        else:
            if self.db_source == 1:
                batch_size, C, H, W = spatial_features_2d_s1.size()

                out = self.cross_att(spatial_features_2d_s1, spatial_features_2d_s1)
                out_s1 = self.se_1(self.dimentionality_upsample(out) + spatial_features_2d_s1)
               
                concat_f = torch.cat([out_s1, spatial_features_2d_s2], 0)
                data_dict['spatial_features_2d'] = concat_f
                
            elif self.db_source==2:
                batch_size, C, H, W = spatial_features_2d_s2.size()

                out = self.cross_att(spatial_features_2d_s2, spatial_features_2d_s2)
                out_s2 = self.se_2(self.dimentionality_upsample(out) + spatial_features_2d_s2)
                
                concat_f = torch.cat([spatial_features_2d_s1, out_s2], 0)
                data_dict['spatial_features_2d'] = concat_f

        return data_dict

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
  
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock_2(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes, stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out

class BasicBlock_Rescale(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_Rescale, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

## similar to channel-wise attention
class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r 
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)

class SA_SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SA_SEBlock, self).__init__()
        self.r = r 
        self.squeeze_1 = nn.Sequential(nn.Conv2d(channels, channels//self.r, 8, 3, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.squeeze_2 = nn.Sequential(nn.Conv2d(channels//self.r, channels//self.r, 8, 3, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.squeeze = nn.Sequential(nn.Linear(channels//self.r, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())
        
    def forward(self, x):
        B, C, H, W = x.size()

        att = self.squeeze_1(x)
        att = self.squeeze_2(att)

        squeeze = self.squeeze(torch.mean(att, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)

    
class DENSE_2D_DT(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.source_one_name = self.model_cfg.SOURCE_ONE_NAME
        self.per_task_channels = self.model_cfg.INPUT_CONV_CHANNEL

        self.db_source = int(self.model_cfg.db_source)

        # SEBlock
        self.se_s1 = SEBlock(self.per_task_channels)
        self.se_s2 = SEBlock(self.per_task_channels)

    def forward(self, data_dict):
        # Get shared representation
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, data_dict)
        spatial_features_2d = data_dict['spatial_features_2d'] 

        spatial_features_2d_s1 = spatial_features_2d[split_tag_s1,:,:,:]
        spatial_features_2d_s2 = spatial_features_2d[split_tag_s2,:,:,:]

        out_s1 = self.se_s1(spatial_features_2d_s1)
        out_s2 = self.se_s2(spatial_features_2d_s2)

        concat_f = torch.cat([out_s1, out_s2], 0)

        data_dict['spatial_features_2d'] = concat_f

        return data_dict

class DENSE_CR(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.N = self.model_cfg.NUM_OF_DB
        self.source_one_name = self.model_cfg.SOURCE_ONE_NAME
        self.per_task_channels = self.model_cfg.INPUT_CONV_CHANNEL
        self.shared_channels = int(self.N*self.model_cfg.INPUT_CONV_CHANNEL)
        self.db_source = int(self.model_cfg.db_source)

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//8, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//8))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//8, downsample=downsample),
                                     nn.Conv2d(self.shared_channels//8, self.shared_channels, 1))

        # Dimensionality reduction 
        self.dimensionality_reduction = BasicBlock_Rescale(self.shared_channels, self.per_task_channels)

        # SEBlock
        self.se_s1 = SEBlock(self.per_task_channels, r=32)
        self.se_s2 = SEBlock(self.per_task_channels, r=32)

    def forward(self, data_dict):
        
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, data_dict)
        spatial_features_2d = data_dict['spatial_features_2d'] 

        spatial_features_2d_s1 = spatial_features_2d[split_tag_s1,:,:,:]
        spatial_features_2d_s2 = spatial_features_2d[split_tag_s2,:,:,:]

        if self.training:
            # Concat the dataset-specific features into the channel-dimension
            concat = torch.cat([spatial_features_2d_s1, spatial_features_2d_s2], 1)
            B, C, H, W = concat.size()
            shared = self.non_linear(concat)

            # Spatial mask across different datasets
            spatial_att = torch.max(concat, dim=1).values.view(B, 1, 1, H, W) 

            # dataset attention mask
            mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim = 2)
            mask = mask * spatial_att
            shared = torch.mul(mask, concat.view(B, C//self.N, self.N, H, W)).view(B,-1, H, W)

            # Perform dimensionality reduction 
            shared = self.dimensionality_reduction(shared)

            # dataset-specific squeeze-and-excitation
            out_s1 = self.se_s1(shared) + spatial_features_2d_s1
            out_s2 = self.se_s2(shared) + spatial_features_2d_s2

            concat_f = torch.cat([out_s1, out_s2], 0)
            data_dict['spatial_features_2d'] = concat_f
        
        else: 
            # Inference Usage: BEV Features Copy
            if self.db_source == 1:
                features_used = spatial_features_2d_s1
            elif self.db_source == 2:
                features_used = spatial_features_2d_s2

            concat = torch.cat([features_used, features_used], 1)
            B, C, H, W = concat.size()
            shared = self.non_linear(concat)

            # Spatial mask across different datasets
            spatial_att = torch.max(concat, dim=1).values.view(B, 1, 1, H, W) 

            # dataset attention mask
            mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim = 2)
            mask = mask * spatial_att
            shared = torch.mul(mask, concat.view(B, C//self.N, self.N, H, W)).view(B,-1, H, W)

            # Perform dimensionality reduction 
            shared = self.dimensionality_reduction(shared)

            # dataset-specific squeeze-and-excitation
            if self.db_source == 1:
                out_s1 = self.se_s1(shared) + spatial_features_2d_s1
                data_dict['spatial_features_2d'] = out_s1
                
            elif self.db_source == 2:
                out_s2 = self.se_s2(shared) + spatial_features_2d_s2
                data_dict['spatial_features_2d'] = out_s2
            
        return data_dict

class DENSE_2D_CR_ADD(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.N = self.model_cfg.NUM_OF_DB
        self.source_one_name = self.model_cfg.SOURCE_ONE_NAME
        self.per_task_channels = self.model_cfg.INPUT_CONV_CHANNEL
        self.shared_channels = int(self.N*self.model_cfg.INPUT_CONV_CHANNEL)
        self.db_source = int(self.model_cfg.db_source)

       # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//4, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//4))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//4, downsample=downsample),
                                     BasicBlock(self.shared_channels//4, self.shared_channels//4),
                                     nn.Conv2d(self.shared_channels//4, self.shared_channels, 1))

        # Dimensionality reduction 
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                    nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels,
                                                    downsample=downsample)

        # SEBlock
        self.se_s1 = SEBlock(self.per_task_channels)
        self.se_s2 = SEBlock(self.per_task_channels)

    def forward(self, data_dict):
        # Get shared representation
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, data_dict)
        spatial_features_2d = data_dict['spatial_features_2d'] 

        spatial_features_2d_s1 = spatial_features_2d[split_tag_s1,:,:,:]
        spatial_features_2d_s2 = spatial_features_2d[split_tag_s2,:,:,:]

        if self.training:
            # Concat the dataset-specific features into the channel-dimension
            concat = torch.cat([spatial_features_2d_s1, spatial_features_2d_s2], 1)
            B, C, H, W = concat.size()
            shared = self.non_linear(concat)

            # dataset attention mask
            mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim = 2)
            shared = torch.mul(mask, concat.view(B, C//self.N, self.N, H, W)).view(B,-1, H, W)

            # Perform dimensionality reduction 
            shared = self.dimensionality_reduction(shared)

            # dataset-specific squeeze-and-excitation
            out_s1 = self.se_s1(shared) + spatial_features_2d_s1
            out_s2 = self.se_s2(shared) + spatial_features_2d_s2

            concat_f = torch.cat([out_s1, out_s2], 0)
            data_dict['spatial_features_2d'] = concat_f
        
        else: 
            # Inference Usage: BEV Features Copy
            if self.db_source == 1:
                features_used = spatial_features_2d_s1
            elif self.db_source == 2:
                features_used = spatial_features_2d_s2

            concat = torch.cat([features_used, features_used], 1)
            B, C, H, W = concat.size()
            shared = self.non_linear(concat)

            # dataset attention mask
            mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim = 2)
            shared = torch.mul(mask, concat.view(B, C//self.N, self.N, H, W)).view(B,-1, H, W)

            # Perform dimensionality reduction 
            shared = self.dimensionality_reduction(shared)

            # dataset-specific squeeze-and-excitation
            if self.db_source == 1:
                out_s1 = self.se_s1(shared) + spatial_features_2d_s1
                data_dict['spatial_features_2d'] = out_s1
            elif self.db_source == 2:
                out_s2 = self.se_s2(shared) + spatial_features_2d_s2
                data_dict['spatial_features_2d'] = out_s2

        return data_dict

class DENSE_2D_CR_ADD_SIM(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.N = self.model_cfg.NUM_OF_DB
        self.source_one_name = self.model_cfg.SOURCE_ONE_NAME
        self.per_task_channels = self.model_cfg.INPUT_CONV_CHANNEL
        self.shared_channels = int(self.N*self.model_cfg.INPUT_CONV_CHANNEL)
        self.db_source = int(self.model_cfg.db_source)

       # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//4, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//4))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//4, downsample=downsample),
                                     BasicBlock(self.shared_channels//4, self.shared_channels//4),
                                     nn.Conv2d(self.shared_channels//4, self.shared_channels, 1))

        # Dimensionality reduction 
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                    nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels,
                                                    downsample=downsample)

        # SEBlock
        self.se_s1 = SEBlock(self.per_task_channels)
        self.se_s2 = SEBlock(self.per_task_channels)

    def forward(self, data_dict):
        # Get shared representation
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, data_dict)
        spatial_features_2d = data_dict['spatial_features_2d'] 

        spatial_features_2d_s1 = spatial_features_2d[split_tag_s1,:,:,:]
        spatial_features_2d_s2 = spatial_features_2d[split_tag_s2,:,:,:]

        # Concat the dataset-specific features into the channel-dimension
        concat = torch.cat([spatial_features_2d_s1, spatial_features_2d_s2], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)

        # dataset attention mask
        mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim = 2)
        shared = torch.mul(mask, concat.view(B, C//self.N, self.N, H, W)).view(B,-1, H, W)

        # Perform dimensionality reduction 
        shared = self.dimensionality_reduction(shared)

        # dataset-specific squeeze-and-excitation
        out_s1 = self.se_s1(shared) + spatial_features_2d_s1
        out_s2 = self.se_s2(shared) + spatial_features_2d_s2

        concat_f = torch.cat([out_s1, out_s2], 0)
        data_dict['spatial_features_2d'] = concat_f

        return data_dict
