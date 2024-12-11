import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from .pointnet import PointNetfeat
from .point_transformer import PointTransformer  # 引入 Point Transformer
from .point_transformer_v3 import PointTransformerV3, Point  # 引入 Point Transformer
from .pointnext import PointNextEncoder  # 引入 PointNext


class PointMultiGraspNet(nn.Module):

    def __init__(self, info_size, k_cls):
        super().__init__()
        self.k_cls = k_cls
        self.pointnet = PointNetfeat(feature_len=3)
        self.point_layer = nn.Sequential(nn.Linear(1024, 512),
                                         nn.LayerNorm(512), nn.Dropout(0.3),
                                         nn.ReLU(inplace=True))
        self.info_layer = nn.Linear(info_size, 32)
        self.anchor_mlp = nn.Sequential(nn.Linear(512 + 32, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, k_cls))
        self.offset_mlp = nn.Sequential(nn.Linear(512 + 32, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, k_cls * 3))

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, points, info):
        # fuse pixel to points
        points = points.transpose(1, 2) #[center-num, 35, 512]
        # pointnet
        features = self.pointnet(points) #[center-num, 1024]
        # mlp
        point_features = self.point_layer(features) #[center-num, 512]
        info_features = self.info_layer(info) #[center-num, 32]
        x = torch.cat([point_features, info_features], 1) #[center-num, 512+32]
        # get anchors and offset
        pred = self.anchor_mlp(x) #[center-num, k_cls]
        offset = self.offset_mlp(x).view(-1, self.k_cls, 3) #[center-num, k_cls, 3]
        return features, pred, offset
    
class PointMultiGraspNet_V3(nn.Module):

    def __init__(self, info_size, k_cls, pretrained_path="dataset_ckpt/point_transformer_v3_pretrain.pth"):
        super().__init__()
        self.k_cls = k_cls
        self.point_transformer = PointTransformerV3(
                in_channels=32,
                order=("z", "z-trans", "hilbert", "hilbert-trans"),
                stride=(2, 2, 2, 2),
                enc_depths=(2, 2, 2, 6, 2),
                enc_channels=(32, 64, 128, 256, 512),
                enc_num_head=(2, 4, 8, 16, 32),
                enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
                drop_path=0.0,
                pre_norm=True,
                shuffle_orders=True,
                enable_rpe=False,
                enable_flash=False,
                upcast_attention=False,
                upcast_softmax=False,
                cls_mode=True,  # 设置为分类模式
                pdnorm_bn=False,
                pdnorm_ln=False,
                pdnorm_decouple=True,
                pdnorm_adaptive=False,
                pdnorm_affine=True,
                pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D")
            )

        self.info_layer = nn.Linear(info_size, 32)
        self.anchor_mlp = nn.Sequential(nn.Linear(512 + 32, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, k_cls))
        self.offset_mlp = nn.Sequential(nn.Linear(512 + 32, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, k_cls * 3))

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Load pretrained weights
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        # Load the pretrained model
        pretrained_model = torch.load(pretrained_path)
        
        # Remove 'state_dict' key if it exists
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']
        
        # Remove 'module.backbone.' prefix from the keys in pretrained_model
        pretrained_dict = {k.replace('module.backbone.', ''): v for k, v in pretrained_model.items()}
        
        # Check if the layers match
        model_dict = self.point_transformer.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        # Update the current model's weights
        model_dict.update(pretrained_dict)
        self.point_transformer.load_state_dict(model_dict)

    def forward(self, points, info):
        # fuse pixel to points
        points = points.transpose(1, 2) #[center-num, 35, 512]
        coord = points[:, :3, :].reshape(-1, 3)  # [center_num * 512, 3]
        feat = points[:, 3:, :].reshape(-1, 32)  # [center_num * 512, 32]
        center_num = points.shape[0]
        device = points.device  # 获取设备
        batch = torch.arange(center_num, device=device).repeat_interleave(512)  # [center_num * 512]
        grid_size = torch.tensor(0.01, device=device)  # 将 grid_size 移动到同一个设备上

        data_dict = {
            "coord": coord,
            "feat": feat,
            "batch": batch,
            "grid_size": grid_size
        }
        point = Point(data_dict)

        # pointnet
        output = self.point_transformer(point)
        # 将特征划分成 [B, K, C] 的大小
        B = len(output.offset)
        K = points.shape[-1]
        C = output.feat.shape[1]

        # 初始化一个形状为 [B, K, C] 的张量
        output_feat = torch.zeros(B, K, C, device="cuda")

        # 填充 output_feat
        start = 0
        for i in range(B):
            end = output.offset[i]
            output_feat[i, :end-start, :] = output.feat[start:end, :]
            start = end

        # 对 K 维度进行最大池化操作
        features = output_feat.max(dim=1)[0] #[center-num, 1024]

        # mlp
        #point_features = self.point_layer(features)
        info_features = self.info_layer(info)
        x = torch.cat([features, info_features], 1)
        # get anchors and offset
        pred = self.anchor_mlp(x)
        offset = self.offset_mlp(x).view(-1, self.k_cls, 3)
        return features, pred, offset
    
class PointMultiGraspNet_PointNext(nn.Module):

    def __init__(self, info_size, k_cls, pretrained_path="dataset_ckpt/scanobjectnn-pointnext-s_best.pth"):
        super().__init__()
        self.k_cls = k_cls
        self.pointnext_encoder = PointNextEncoder(
            in_channels=4,  # 输入点云的通道数，例如4表示(x, y, z, intensity)
            width=32,
            blocks=[1, 1, 1, 1, 1, 1],
            strides=[1, 2, 2, 2, 2, 1],
            sa_layers=2,
            sa_use_res=True,
            expansion=4,
            radius=0.15,
            radius_scaling=1.5,
            nsample=32,
            aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
            group_args={'NAME': 'ballquery', 'normalize_dp': True},
            conv_args={'order': 'conv-norm-act'},
            act_args={'act': 'relu'},
            norm_args={'norm': 'bn'}
        )
        self.info_layer = nn.Linear(info_size, 32)
        self.anchor_mlp = nn.Sequential(nn.Linear(512 + 32, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, k_cls))
        self.offset_mlp = nn.Sequential(nn.Linear(512 + 32, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, k_cls * 3))
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Load pretrained weights
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        # Load the pretrained model
        pretrained_model = torch.load(pretrained_path)
        
        # Remove 'state_dict' key if it exists
        if 'model' in pretrained_model:
            pretrained_model = pretrained_model['model']
        
        # Remove 'module.backbone.' prefix from the keys in pretrained_model
        pretrained_dict = {k.replace('encoder.encoder.', 'encoder.'): v for k, v in pretrained_model.items()}
        
        # Check if the layers match
        model_dict = self.pointnext_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        # Update the current model's weights
        model_dict.update(pretrained_dict)
        self.pointnext_encoder.load_state_dict(model_dict)
        print("Pretrained weights loaded successfully!")

    def forward(self, points, info):
        # fuse pixel to points
        points = points.transpose(1, 2) #[center-num, 35, 512]
        p0 = points[ : , :3 ,: ].transpose(1, 2).contiguous()  # 取前3个通道作为位置[center-num, 512, 3]
        f0 = points[:, 3:, :].max(dim=1, keepdim=True)[0] # 最大池化到[center-num, 1, 512]
        f0 = torch.cat([points[:, :3, :], f0], dim=1).contiguous() # 拼接到[center-num, 4, 512]
        # 使用 PointNext 编码器
        features = self.pointnext_encoder.forward_cls_feat(p0, f0) #[center_num, 512, 2]

        # mlp
        info_features = self.info_layer(info)
        x = torch.cat([features, info_features], 1)
        # get anchors and offset
        pred = self.anchor_mlp(x)
        offset = self.offset_mlp(x).view(-1, self.k_cls, 3)
        return features, pred, offset
