#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:04:06 2022

@author: leeh43
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F
from nnunetv2.custom_networks.UXNet_3D.uxnet_encoder import uxnet_conv
import functools
import os
import pdb
from torch.nn.functional import interpolate
import torch
import torch.nn as nn

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


class ModuleHelper(object):
    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type == "torchbn":
            return nn.Sequential(nn.BatchNorm3d(num_features, **kwargs), nn.ReLU())
        elif bn_type == "torchsyncbn":
            return nn.Sequential(nn.SyncBatchNorm(num_features, **kwargs), nn.ReLU())
        elif bn_type == "syncbn":
            from lib.extensions.syncbn.module import BatchNorm2d

            return nn.Sequential(BatchNorm2d(num_features, **kwargs), nn.ReLU())
        elif bn_type == "sn":
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d

            return nn.Sequential(SwitchNorm2d(num_features, **kwargs), nn.ReLU())
        elif bn_type == "gn":
            return nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=num_features, **kwargs),
                nn.ReLU(),
            )
        elif bn_type == "fn":
            print("Not support Filter-Response-Normalization: {}.".format(bn_type))
            exit(1)
        elif bn_type == "inplace_abn":
            torch_ver = torch.__version__[:3]
            # print('Pytorch Version: {}'.format(torch_ver))
            if torch_ver == "0.4":
                from lib.extensions.inplace_abn.bn import InPlaceABNSync

                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver in ("1.0", "1.1"):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync

                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver == "1.2":
                from inplace_abn import InPlaceABNSync

                return InPlaceABNSync(num_features, **kwargs)

        else:
            print("Not support BN type: {}.".format(bn_type))
            exit(1)

    @staticmethod
    def BatchNorm2d(bn_type="torch", ret_cls=False):
        if bn_type == "torchbn":
            return nn.BatchNorm2d

        elif bn_type == "torchsyncbn":
            return nn.SyncBatchNorm

        elif bn_type == "syncbn":
            from lib.extensions.syncbn.module import BatchNorm2d

            return BatchNorm2d

        elif bn_type == "sn":
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d

            return SwitchNorm2d

        elif bn_type == "gn":
            return functools.partial(nn.GroupNorm, num_groups=32)

        elif bn_type == "inplace_abn":
            torch_ver = torch.__version__[:3]
            if torch_ver == "0.4":
                from lib.extensions.inplace_abn.bn import InPlaceABNSync

                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation="none")

            elif torch_ver in ("1.0", "1.1"):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync

                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation="none")

            elif torch_ver == "1.2":
                from inplace_abn import InPlaceABNSync

                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation="identity")

        else:
            print("Not support BN type: {}.".format(bn_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network="resnet101"):
        if pretrained is None:
            return model

        if all_match:
            print("Loading pretrained model:{}".format(pretrained))
            pretrained_dict = torch.load(
                pretrained, map_location=lambda storage, loc: storage
            )
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if "resinit.{}".format(k) in model_dict:
                    load_dict["resinit.{}".format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)

        else:
            print("Loading pretrained model:{}".format(pretrained))
            pretrained_dict = torch.load(
                pretrained, map_location=lambda storage, loc: storage
            )

            # settings for "wide_resnet38"  or network == "resnet152"
            if network == "wide_resnet":
                pretrained_dict = pretrained_dict["state_dict"]

            model_dict = model.state_dict()

            if network == "hrnet_plus":
                # pretrained_dict['conv1_full_res.weight'] = pretrained_dict['conv1.weight']
                # pretrained_dict['conv2_full_res.weight'] = pretrained_dict['conv2.weight']
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }

            elif network == "pvt":
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }
                pretrained_dict["pos_embed1"] = interpolate(
                    pretrained_dict["pos_embed1"].unsqueeze(dim=0), size=[16384, 64]
                )[0]
                pretrained_dict["pos_embed2"] = interpolate(
                    pretrained_dict["pos_embed2"].unsqueeze(dim=0), size=[4096, 128]
                )[0]
                pretrained_dict["pos_embed3"] = interpolate(
                    pretrained_dict["pos_embed3"].unsqueeze(dim=0), size=[1024, 320]
                )[0]
                pretrained_dict["pos_embed4"] = interpolate(
                    pretrained_dict["pos_embed4"].unsqueeze(dim=0), size=[256, 512]
                )[0]
                pretrained_dict["pos_embed7"] = interpolate(
                    pretrained_dict["pos_embed1"].unsqueeze(dim=0), size=[16384, 64]
                )[0]
                pretrained_dict["pos_embed6"] = interpolate(
                    pretrained_dict["pos_embed2"].unsqueeze(dim=0), size=[4096, 128]
                )[0]
                pretrained_dict["pos_embed5"] = interpolate(
                    pretrained_dict["pos_embed3"].unsqueeze(dim=0), size=[1024, 320]
                )[0]
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }

            elif network == "pcpvt" or network == "svt":
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }
                print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))

            elif network == "transunet_swin":
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }
                for item in list(pretrained_dict.keys()):
                    if item.startswith("layers.0") and not item.startswith(
                        "layers.0.downsample"
                    ):
                        pretrained_dict["dec_layers.2" + item[15:]] = pretrained_dict[
                            item
                        ]
                    if item.startswith("layers.1") and not item.startswith(
                        "layers.1.downsample"
                    ):
                        pretrained_dict["dec_layers.1" + item[15:]] = pretrained_dict[
                            item
                        ]
                    if item.startswith("layers.2") and not item.startswith(
                        "layers.2.downsample"
                    ):
                        pretrained_dict["dec_layers.0" + item[15:]] = pretrained_dict[
                            item
                        ]

                for item in list(pretrained_dict.keys()):
                    if "relative_position_index" in item:
                        pretrained_dict[item] = interpolate(
                            pretrained_dict[item]
                            .unsqueeze(dim=0)
                            .unsqueeze(dim=0)
                            .float(),
                            size=[256, 256],
                        )[0][0]
                    if "relative_position_bias_table" in item:
                        pretrained_dict[item] = interpolate(
                            pretrained_dict[item]
                            .unsqueeze(dim=0)
                            .unsqueeze(dim=0)
                            .float(),
                            size=[961, pretrained_dict[item].size(1)],
                        )[0][0]
                    if "attn_mask" in item:
                        pretrained_dict[item] = interpolate(
                            pretrained_dict[item]
                            .unsqueeze(dim=0)
                            .unsqueeze(dim=0)
                            .float(),
                            size=[pretrained_dict[item].size(0), 256, 256],
                        )[0][0]

            elif network == "hrnet" or network == "xception" or network == "resnest":
                load_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
                }
                print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))

            elif network == "dcnet" or network == "resnext":
                load_dict = dict()
                for k, v in pretrained_dict.items():
                    if "resinit.{}".format(k) in model_dict:
                        load_dict["resinit.{}".format(k)] = v
                    else:
                        if k in model_dict:
                            load_dict[k] = v
                        else:
                            pass

            elif network == "wide_resnet":
                load_dict = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in pretrained_dict.items()
                    if ".".join(k.split(".")[1:]) in model_dict
                }
            else:
                load_dict = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in pretrained_dict.items()
                    if ".".join(k.split(".")[1:]) in model_dict
                }

            # used to debug
            if int(os.environ.get("debug_load_model", 0)):
                print("Matched Keys List:")
                for key in load_dict.keys():
                    print("{}".format(key))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join("~", ".PyTorchCV", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split("/")[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        print("Loading pretrained model:{}".format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution="normal"):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(
        module, mode="fan_in", nonlinearity="leaky_relu", bias=0, distribution="normal"
    ):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj="convmlp", bn_type="torchbn"):
        super(ProjectionHead, self).__init__()

        print("proj_dim: {}".format(proj_dim))

        if proj == "linear":
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == "convmlp":
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1),
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


# class ResBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,
#                  in_planes: int,
#                  planes: int,
#                  spatial_dims: int = 3,
#                  stride: int = 1,
#                  downsample: Union[nn.Module, partial, None] = None,
#     ) -> None:
#         """
#         Args:
#             in_planes: number of input channels.
#             planes: number of output channels.
#             spatial_dims: number of spatial dimensions of the input image.
#             stride: stride to use for first conv layer.
#             downsample: which downsample layer to use.
#         """
#
#         super().__init__()
#
#         conv_type: Callable = Conv[Conv.CONV, spatial_dims]
#         norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
#
#         self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.bn1 = norm_type(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = norm_type(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         residual = x
#
#         out: torch.Tensor = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class UXNET(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        # self.classification = False
        # self.vit = ViT(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        # )
        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans
        )
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        # print(outs[1].size())
        # print(outs[2].size())
        # print(outs[3].size())
        enc1 = self.encoder1(x_in)
        # print(enc1.size())
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        # print(enc2.size())
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        # print(enc3.size())
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        # print(enc4.size())
        # dec4 = self.proj_feat(outs[3], self.hidden_size, self.feat_size)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        # feat = self.conv_proj(dec4)

        return self.out(out)
