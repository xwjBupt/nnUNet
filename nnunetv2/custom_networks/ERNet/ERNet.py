import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import pdb
from nnunetv2.custom_networks.ERNet.components import (
    BasicUp,
    BasicDown,
    conv1x1,
    conv3x3,
    VAEBasicUp,
    ASPP,
    PPM,
    encoder,
    SELayer,
)
from nnunetv2.custom_networks.ERNet.VIT import Baseformer


class ERNet(nn.Module):
    def __init__(
        self,
        in_ch_seg=1,
        out_ch_seg=1,
        middle_out=[32, 64, 128, 256, 512],
        deep_supervision=True,
        deconv=False,
        norm_layer=None,
        activation=None,
        trans_mode="baseformerV2",
        loop=1,
        **kwargs
    ):
        super(ERNet, self).__init__()
        self.deep_supervision = deep_supervision
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = nn.SyncBatchNorm

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        self.loop = loop

        self.in_ch_seg = in_ch_seg
        self.out_ch_seg = out_ch_seg
        self.in_ch_edge = in_ch_seg
        self.out_ch_edge = out_ch_seg
        self.trans_mode = trans_mode

        self.init_conv_seg = nn.Sequential(
            conv3x3(self.in_ch_seg, middle_out[0]),
            self.norm_layer(middle_out[0]),
            self.activation,
        )

        self.ENCODER = encoder(
            middle_out[0],
            middle_out[1:],
            activation=self.activation,
            norm_layer=self.norm_layer,
        )

        # self.decode1_seg = BasicUp(middle_out[-1], middle_out[-2], deconv, norm_layer=self.norm_layer,
        #                            activation=self.activation)
        # self.decode2_seg = BasicUp(middle_out[-2], middle_out[-3], deconv, norm_layer=self.norm_layer,
        #                            activation=self.activation)
        # self.decode3_seg = BasicUp(middle_out[-3], middle_out[-4], deconv, norm_layer=self.norm_layer,
        #                            activation=self.activation)
        # self.decode4_seg = BasicUp(middle_out[-4], middle_out[-5], deconv, norm_layer=self.norm_layer,
        #
        self.flat_erode = conv1x1(middle_out[-1], middle_out[-1])
        self.flat_dilate = conv1x1(middle_out[-1], middle_out[-1])
        self.flat_seg = conv1x1(middle_out[-1], middle_out[-1])

        self.seg_up_4 = conv3x3(middle_out[-1], middle_out[-2])
        self.seg_up_3 = conv3x3(middle_out[-2], middle_out[-3])
        self.seg_up_2 = conv3x3(middle_out[-3], middle_out[-4])
        self.seg_up_1 = conv3x3(middle_out[-4], middle_out[-5])

        self.decode4_seg = BasicDown(
            middle_out[-2],
            middle_out[-2],
            stride=1,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode3_seg = BasicDown(
            middle_out[-3],
            middle_out[-3],
            stride=1,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode2_seg = BasicDown(
            middle_out[-4],
            middle_out[-4],
            stride=1,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode1_seg = BasicDown(
            middle_out[-5],
            middle_out[-5],
            stride=1,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )

        self.decode4_erode = BasicUp(
            middle_out[-1],
            middle_out[-2],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode3_erode = BasicUp(
            middle_out[-2],
            middle_out[-3],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode2_erode = BasicUp(
            middle_out[-3],
            middle_out[-4],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode1_erode = BasicUp(
            middle_out[-4],
            middle_out[-5],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )

        self.decode4_dilate = BasicUp(
            middle_out[-1],
            middle_out[-2],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode3_dilate = BasicUp(
            middle_out[-2],
            middle_out[-3],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode2_dilate = BasicUp(
            middle_out[-3],
            middle_out[-4],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode1_dilate = BasicUp(
            middle_out[-4],
            middle_out[-5],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )

        self.final_conv_seg = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv3x3(middle_out[-5], middle_out[-5])),
                    ("bn1", self.norm_layer(middle_out[-5])),
                    (
                        "activation",
                        self.activation,
                    ),
                ]
            )
        )
        self.outconv_seg = conv1x1(middle_out[-5], self.out_ch_seg)

        self.final_conv_erode = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv3x3(middle_out[-5], middle_out[-5])),
                    ("bn1", self.norm_layer(middle_out[-5])),
                    (
                        "activation",
                        self.activation,
                    ),
                ]
            )
        )
        self.outconv_erode = conv1x1(middle_out[-5], self.out_ch_seg)

        self.erode_trans = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv3x3(middle_out[-5], middle_out[-5])),
                    ("bn1", self.norm_layer(middle_out[-5])),
                    (
                        "activation",
                        self.activation,
                    ),
                ]
            )
        )

        self.final_conv_dilate = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv3x3(middle_out[-5], middle_out[-5])),
                    ("bn1", self.norm_layer(middle_out[-5])),
                    (
                        "activation",
                        self.activation,
                    ),
                ]
            )
        )
        self.outconv_dilate = conv1x1(middle_out[-5], self.out_ch_seg)

        self.dilate_trans = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv3x3(middle_out[-5], middle_out[-5])),
                    ("bn1", self.norm_layer(middle_out[-5])),
                    (
                        "activation",
                        self.activation,
                    ),
                ]
            )
        )

        self.trans1 = Transfer(
            middle_out[0],
            patch_dim=65536,
            trans_mode=self.trans_mode,
            activation=self.activation,
            norm_layer=self.norm_layer,
            loop=self.loop,
            **kwargs
        )
        self.trans2 = Transfer(
            middle_out[1],
            patch_dim=16384,
            trans_mode=self.trans_mode,
            activation=self.activation,
            norm_layer=self.norm_layer,
            loop=self.loop,
            **kwargs
        )
        self.trans3 = Transfer(
            middle_out[2],
            patch_dim=4096,
            trans_mode=self.trans_mode,
            activation=self.activation,
            norm_layer=self.norm_layer,
            loop=self.loop,
            **kwargs
        )
        self.trans4 = Transfer(
            middle_out[3],
            patch_dim=1024,
            trans_mode=self.trans_mode,
            activation=self.activation,
            norm_layer=self.norm_layer,
            loop=self.loop,
            **kwargs
        )

        if True:
            self.side2_seg_conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv1",
                            BasicDown(
                                middle_out[-3],
                                middle_out[-5],
                                stride=1,
                                norm_layer=self.norm_layer,
                                activation=self.activation,
                            ),
                        ),
                        ("conv2", conv1x1(middle_out[-5], self.out_ch_seg)),
                    ]
                )
            )
            self.side2_erode_conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv1",
                            BasicDown(
                                middle_out[-3],
                                middle_out[-5],
                                stride=1,
                                norm_layer=self.norm_layer,
                                activation=self.activation,
                            ),
                        ),
                        ("conv2", conv1x1(middle_out[-5], self.out_ch_seg)),
                    ]
                )
            )

            self.side2_dilate_conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv1",
                            BasicDown(
                                middle_out[-3],
                                middle_out[-5],
                                stride=1,
                                norm_layer=self.norm_layer,
                                activation=self.activation,
                            ),
                        ),
                        ("conv2", conv1x1(middle_out[-5], self.out_ch_seg)),
                    ]
                )
            )

    def forward(self, img):
        seg_init = self.init_conv_seg(img)
        x1_seg, x2_seg, x3_seg, x4_seg = self.ENCODER(seg_init)

        x4_erode = self.flat_erode(x4_seg)
        x4_dilate = self.flat_dilate(x4_seg)
        x4_seg = self.flat_seg(x4_seg)

        x3_dilate = self.decode4_dilate(x4_dilate, x3_seg)
        x3_erode = self.decode4_erode(x4_erode, x3_seg)
        x3_seg = self.trans4(
            self.seg_up_4(self.upsample(x4_seg)) + x3_seg, x3_erode, x3_dilate
        )
        x3_seg = self.decode4_seg(x3_seg)

        x2_dilate = self.decode3_dilate(x3_dilate, x2_seg)
        x2_erode = self.decode3_erode(x3_erode, x2_seg)
        x2_seg = self.trans3(
            self.seg_up_3(self.upsample(x3_seg)) + x2_seg, x2_erode, x2_dilate
        )
        x2_seg = self.decode3_seg(x2_seg)

        x1_dilate = self.decode2_dilate(x2_dilate, x1_seg)
        x1_erode = self.decode2_erode(x2_erode, x1_seg)
        x1_seg = self.trans2(
            self.seg_up_2(self.upsample(x2_seg)) + x1_seg, x1_erode, x1_dilate
        )
        x1_seg = self.decode2_seg(x1_seg)

        x0_dilate = self.decode1_dilate(x1_dilate, seg_init)
        x0_erode = self.decode1_erode(x1_erode, seg_init)
        x0_seg = self.trans1(
            self.seg_up_1(self.upsample(x1_seg)) + seg_init, x0_erode, x0_dilate
        )
        x0_seg = self.decode1_seg(x0_seg)

        x0_dilate = self.final_conv_dilate(x0_dilate + seg_init)
        x0_erode = self.final_conv_erode(x0_erode + seg_init)

        ### kernel ###
        x0_seg = x0_seg + self.erode_trans(x0_erode) - self.dilate_trans(x0_dilate)
        ### kernel ###
        x0_seg = self.final_conv_seg(x0_seg)

        x0_seg = self.outconv_seg(x0_seg)
        x0_dilate = self.outconv_dilate(x0_dilate)
        x0_erode = self.outconv_erode(x0_erode)
        if self.deep_supervision:
            x2_seg = self.side2_seg_conv(x2_seg)
            x2_erode = self.side2_erode_conv(x2_erode)
            x2_dilate = self.side2_dilate_conv(x2_dilate)
            return [x0_seg, x0_erode, x0_dilate, x2_seg, x2_erode, x2_dilate]
        else:
            # return [x0_seg, x0_erode, x0_dilate]
            return x0_seg

    def upsample(self, x, times=2, mode="bilinear"):
        x = F.interpolate(x, scale_factor=times, mode=mode)
        return x


class ERNet_Pruned(nn.Module):
    def __init__(
        self,
        in_ch_seg=1,
        out_ch_seg=1,
        in_ch_edge=1,
        out_ch_edge=1,
        middle_out=[32, 64, 128, 256, 512],
        deconv=False,
        norm_layer=None,
        activation=None,
        trans_mode="plus",
        expand="dilate",
        loop=1,
        **kwargs
    ):
        super(ERNet_Pruned, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = nn.SyncBatchNorm

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        self.loop = loop
        self.expand = expand
        self.in_ch_seg = in_ch_seg
        self.out_ch_seg = out_ch_seg
        self.in_ch_edge = in_ch_seg
        self.out_ch_edge = out_ch_seg
        self.trans_mode = trans_mode

        self.init_conv_seg = nn.Sequential(
            conv3x3(self.in_ch_seg, middle_out[0]),
            self.norm_layer(middle_out[0]),
            nn.LeakyReLU(inplace=True),
        )

        self.ENCODER = encoder(
            middle_out[0],
            middle_out[1:],
            activation=self.activation,
            norm_layer=self.norm_layer,
        )

        # self.decode1_seg = BasicUp(middle_out[-1], middle_out[-2], deconv, norm_layer=self.norm_layer,
        #                            activation=self.activation)
        # self.decode2_seg = BasicUp(middle_out[-2], middle_out[-3], deconv, norm_layer=self.norm_layer,
        #                            activation=self.activation)
        # self.decode3_seg = BasicUp(middle_out[-3], middle_out[-4], deconv, norm_layer=self.norm_layer,
        #                            activation=self.activation)
        # self.decode4_seg = BasicUp(middle_out[-4], middle_out[-5], deconv, norm_layer=self.norm_layer,

        self.flat_erode = conv1x1(middle_out[-1], middle_out[-1])
        self.flat_dilate = conv1x1(middle_out[-1], middle_out[-1])
        self.flat_seg = conv1x1(middle_out[-1], middle_out[-1])

        self.seg_up_4 = conv3x3(middle_out[-1], middle_out[-2])
        self.seg_up_3 = conv3x3(middle_out[-2], middle_out[-3])
        self.seg_up_2 = conv3x3(middle_out[-3], middle_out[-4])
        self.seg_up_1 = conv3x3(middle_out[-4], middle_out[-5])

        self.decode4_seg = BasicDown(
            middle_out[-2],
            middle_out[-2],
            stride=1,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode3_seg = BasicDown(
            middle_out[-3],
            middle_out[-3],
            stride=1,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode2_seg = BasicDown(
            middle_out[-4],
            middle_out[-4],
            stride=1,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode1_seg = BasicDown(
            middle_out[-5],
            middle_out[-5],
            stride=1,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )

        self.decode4_expand = BasicUp(
            middle_out[-1],
            middle_out[-2],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode3_expand = BasicUp(
            middle_out[-2],
            middle_out[-3],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode2_expand = BasicUp(
            middle_out[-3],
            middle_out[-4],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )
        self.decode1_expand = BasicUp(
            middle_out[-4],
            middle_out[-5],
            deconv,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )

        self.final_conv_seg = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv3x3(middle_out[-5], middle_out[-5])),
                    ("bn1", self.norm_layer(middle_out[-5])),
                    (
                        "activation",
                        nn.LeakyReLU(inplace=True),
                    ),
                ]
            )
        )
        self.outconv_seg = conv1x1(middle_out[-5], self.out_ch_seg)

        self.final_conv_expand = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv3x3(middle_out[-5], middle_out[-5])),
                    ("bn1", self.norm_layer(middle_out[-5])),
                    (
                        "activation",
                        nn.LeakyReLU(inplace=True),
                    ),
                ]
            )
        )
        self.outconv_expand = conv1x1(middle_out[-5], self.out_ch_seg)

        self.trans1 = Transfer(
            middle_out[0],
            self.expand,
            self.trans_mode,
            activation=self.activation,
            norm_layer=self.norm_layer,
            loop=self.loop,
            **kwargs
        )
        self.trans2 = Transfer(
            middle_out[1],
            self.expand,
            self.trans_mode,
            activation=self.activation,
            norm_layer=self.norm_layer,
            loop=self.loop,
            **kwargs
        )
        self.trans3 = Transfer(
            middle_out[2],
            self.expand,
            self.trans_mode,
            activation=self.activation,
            norm_layer=self.norm_layer,
            loop=self.loop,
            **kwargs
        )
        self.trans4 = Transfer(
            middle_out[3],
            self.expand,
            self.trans_mode,
            activation=self.activation,
            norm_layer=self.norm_layer,
            loop=self.loop,
            **kwargs
        )

    def forward(self, img):
        seg_init = self.init_conv_seg(img)
        x1_seg, x2_seg, x3_seg, x4_seg = self.ENCODER(seg_init)

        x4_expand = self.flat_erode(x4_seg)
        x4_seg = self.flat_seg(x4_seg)

        x3_expand = self.decode4_expand(x4_expand, x3_seg)
        x3_seg = self.trans4(self.seg_up_4(self.upsample(x4_seg)), x3_expand, x3_expand)
        x3_seg = self.decode4_seg(x3_seg)

        x2_expand = self.decode3_expand(x3_expand, x2_seg)
        x2_seg = self.trans3(self.seg_up_3(self.upsample(x3_seg)), x2_expand, x2_expand)
        x2_seg = self.decode3_seg(x2_seg)

        x1_expand = self.decode2_expand(x2_expand, x1_seg)
        x1_seg = self.trans2(self.seg_up_2(self.upsample(x2_seg)), x1_expand, x1_expand)
        x1_seg = self.decode2_seg(x1_seg)

        x0_expand = self.decode1_expand(x1_expand, seg_init)
        x0_seg = self.trans1(self.seg_up_1(self.upsample(x1_seg)), x0_expand, x0_expand)
        x0_seg = self.decode1_seg(x0_seg)

        x0_expand = self.final_conv_expand(x0_expand + seg_init)

        ### kernel ###
        if self.expand == "dilate":
            x0_seg = x0_seg - x0_expand
        else:
            x0_seg = x0_seg + x0_expand
        ### kernel ###
        x0_seg = self.final_conv_seg(x0_seg)
        x0_seg = self.outconv_seg(x0_seg)
        x0_expand = self.outconv_expand(x0_expand)

        return [x0_seg, x0_expand, x0_expand]

    def upsample(self, x, times=2, mode="bilinear"):
        x = F.interpolate(x, scale_factor=times, mode=mode)
        return x


class Transfer(nn.Module):
    def __init__(
        self,
        ch,
        patch_dim=None,
        in_dim=1024,
        trans_mode=None,
        expand="norm",
        norm_layer=None,
        activation=None,
        loop=1,
        **kwargs
    ):
        super(Transfer, self).__init__()
        self.trans_mode = trans_mode
        self.expand = expand
        assert self.expand in ["dilate", "erode", "norm"], "not supprot type"
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = norm_layer

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation

        if self.trans_mode == "plus":
            # self.fuse = conv3x3(ch, ch)
            self.fuse = nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", conv3x3(ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )
        elif self.trans_mode == "concat":
            self.erode_suppress = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_erode_suppress", conv1x1(ch, ch // 3)),
                        ("bn1", self.norm_layer(ch // 3)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )

            self.dilate_suppress = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_dilate_suppress", conv1x1(ch, ch // 3)),
                        ("bn1", self.norm_layer(ch // 3)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )

            self.seg_suppress = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_seg_suppress", conv1x1(ch, ch // 3)),
                        ("bn1", self.norm_layer(ch // 3)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )

            self.f1 = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_fuse_1x1", conv1x1(ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )

            self.f2 = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_fuse_1x1", conv3x3(ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )

            self.f3 = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_fuse_3x3_d2", conv3x3(ch, ch, dilation=2)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                        ("conv_fuse_1x1", conv3x3(ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )

            self.fuse = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_fuse_1x1_final", conv3x3(3 * ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )
        elif self.trans_mode == "baseformer":
            self.bv = Baseformer(
                ch=ch,
                dim=in_dim,
                patch_dim=patch_dim,
                # depth=1,
                # heads=16,
                # mlp_dim=2048,
                # dropout=0.1,
                **kwargs
            )
            self.erode_trans = nn.Sequential(
                OrderedDict(
                    [
                        ("erode_trans", conv1x1(ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )

            self.dilate_trans = nn.Sequential(
                OrderedDict(
                    [
                        ("dilate_trans", conv1x1(ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )
        elif self.trans_mode == "baseformerV2":
            self.bv = Baseformer(
                ch=ch,
                dim=in_dim,
                patch_dim=patch_dim,
                # depth=1,
                # heads=16,
                # mlp_dim=2048,
                # dropout=0.1,
                **kwargs
            )
            self.erode_trans = nn.Sequential(
                OrderedDict(
                    [
                        ("erode_trans", conv1x1(ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )

            self.dilate_trans = nn.Sequential(
                OrderedDict(
                    [
                        ("dilate_trans", conv1x1(ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                    ]
                )
            )
            self.concat_fuse = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_fuse_3x3", conv3x3(3 * ch, ch)),
                        ("bn1", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                        ("conv_fuse_3x3_d2", conv3x3(ch, ch, dilation=2)),
                        ("bn2", self.norm_layer(ch)),
                        (
                            "activation",
                            self.activation,
                        ),
                        ("conv_fuse_1x1", conv1x1(ch, ch)),
                        ("bn3", self.norm_layer(ch)),
                    ]
                )
            )

    def forward(self, seg, erode, dilate):
        if self.trans_mode is None:
            return seg
        elif self.trans_mode == "plus":
            if self.expand == "norm":
                seg = seg + F.softmax(erode, dim=1) - F.softmax(dilate, dim=1)
                seg = self.fuse(seg)
            elif self.expand == "dilate":
                seg = seg - dilate
            elif self.expand == "erode":
                seg = seg + erode
        elif self.trans_mode == "concat":
            es = seg + F.softmax(erode, dim=1)
            ds = seg - F.softmax(dilate, dim=1)

            es = self.erode_suppress(es)
            seg = self.seg_suppress(seg)
            ds = self.dilate_suppress(ds)

            seg = torch.cat([es, seg, ds], dim=1)
            seg_f1 = self.f1(seg)
            seg_f2 = self.f2(seg)
            seg_f3 = self.f3(seg)

            seg = self.fuse(torch.cat([seg_f1, seg_f2, seg_f3], dim=1))
        elif self.trans_mode == "baseformer":
            erode = self.erode_trans(erode)
            dilate = self.dilate_trans(dilate)
            # seg = seg + erode - dilate
            seg = seg + F.softmax(erode, dim=1) - F.softmax(dilate, dim=1)
            seg = self.bv(seg)
        elif self.trans_mode == "baseformerV2":
            erode = self.erode_trans(erode)
            dilate = self.dilate_trans(dilate)
            softed_erode = F.softmax(erode, dim=1)
            softed_dilate = F.softmax(dilate, dim=1)
            # seg = seg + erode - dilate
            seg1 = seg + softed_erode - softed_dilate
            seg1 = self.bv(seg1)

            seg2 = torch.cat([seg, seg + softed_erode, seg - softed_dilate], dim=1)
            seg2 = self.concat_fuse(seg2)

            seg = self.activation(seg1 + seg + seg2)
        else:
            assert False, "not surpport"
        return seg


if __name__ == "__main__":
    seg = torch.randn([4, 1, 512, 512])
    net = ERNet(trans_mode="baseformerV2", depth=1, heads=16, dropout=0.1, mlp_dim=2048)
    print(net)
    out = net(seg)
    print(out)
