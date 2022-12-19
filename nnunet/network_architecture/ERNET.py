from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.initialization import InitWeights_He
from torch import nn, Tensor
from typing import Optional, List
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import pdb
from torchstat import stat


class encoder(nn.Module):
    def __init__(
        self, in_ch, out_ch=[64, 128, 256, 512], norm_layer=None, activation=None
    ):
        super(encoder, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = norm_layer

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation

        self.encode_stage1 = BasicDown(
            in_ch,
            out_ch[0],
            stride=2,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )  # 1/2
        self.encode_stage2 = BasicDown(
            out_ch[0],
            out_ch[1],
            stride=2,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )  # 1/4
        self.encode_stage3 = BasicDown(
            out_ch[1],
            out_ch[2],
            stride=2,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )  # 1/8
        self.encode_stage4 = BasicDown(
            out_ch[2],
            out_ch[3],
            stride=2,
            norm_layer=self.norm_layer,
            activation=self.activation,
        )  # 1/16

    def forward(self, x):

        x1 = self.encode_stage1(x)
        x2 = self.encode_stage2(x1)
        x3 = self.encode_stage3(x2)
        x4 = self.encode_stage4(x3)
        return x1, x2, x3, x4


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv3x3(in_planes, out_planes):
    return nn.ConvTranspose2d(in_planes, out_planes, 2, stride=2)


class BasicDown(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicDown, self).__init__()

        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        else:
            self._norm_layer = norm_layer

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation

        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = self._norm_layer(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = self._norm_layer(out_ch)

        self.stride = stride
        if stride > 1 or out_ch != in_ch:
            self.downsample = nn.Sequential(
                conv1x1(in_ch, out_ch, stride),
                self._norm_layer(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class BasicUp(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        deconv: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicUp, self).__init__()
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        else:
            self._norm_layer = norm_layer

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation
        self.deconv = deconv

        if self.deconv:
            self.fuse = deconv3x3(in_ch, out_ch)
        else:
            self.fuse = conv3x3(in_ch, out_ch)

        self.upbn = self._norm_layer(out_ch)
        self.conv1 = conv3x3(out_ch, out_ch)
        self.bn1 = self._norm_layer(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = self._norm_layer(out_ch)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if self.deconv:
            x = self.fuse(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.fuse(x)
        x = self.upbn(x)
        x = x + skip
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.activation(x)
        return x


class ERNet(nn.Module):
    def __init__(
        self,
        in_ch_seg=1,
        out_ch_seg=1,
        in_ch_edge=1,
        out_ch_edge=1,
        middle_out=[32, 64, 128, 256, 512],
        deep_supervison=False,
        deconv=False,
        norm_layer=None,
        activation=None,
        trans_mode="plus",
        loop=1,
        **kwargs
    ):
        super(ERNet, self).__init__()
        self.deep_supervison = deep_supervison
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

        if self.deep_supervison:
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
        x3_seg = self.trans4(self.seg_up_4(self.upsample(x4_seg)), x3_erode, x3_dilate)
        x3_seg = self.decode4_seg(x3_seg)

        x2_dilate = self.decode3_dilate(x3_dilate, x2_seg)
        x2_erode = self.decode3_erode(x3_erode, x2_seg)
        x2_seg = self.trans3(self.seg_up_3(self.upsample(x3_seg)), x2_erode, x2_dilate)
        x2_seg = self.decode3_seg(x2_seg)

        x1_dilate = self.decode2_dilate(x2_dilate, x1_seg)
        x1_erode = self.decode2_erode(x2_erode, x1_seg)
        x1_seg = self.trans2(self.seg_up_2(self.upsample(x2_seg)), x1_erode, x1_dilate)
        x1_seg = self.decode2_seg(x1_seg)

        x0_dilate = self.decode1_dilate(x1_dilate, seg_init)
        x0_erode = self.decode1_erode(x1_erode, seg_init)
        x0_seg = self.trans1(self.seg_up_1(self.upsample(x1_seg)), x0_erode, x0_dilate)
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
        if self.deep_supervison:
            x2_seg = self.side2_seg_conv(x2_seg)
            x2_erode = self.side2_erode_conv(x2_erode)
            x2_dilate = self.side2_dilate_conv(x2_dilate)
            return [x0_seg, x0_dilate, x0_erode, x2_seg, x2_dilate, x2_erode]
        else:
            return [x0_seg, x0_dilate, x0_erode]

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


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(
            torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True
        )
        self.rel_w = nn.Parameter(
            torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (
            (self.rel_h + self.rel_w)
            .view(1, self.heads, C // self.heads, -1)
            .permute(0, 1, 3, 2)
        )
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                **kwargs
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Baseformer(nn.Module):
    def __init__(
        self,
        ch,
        patch_dim,
        dim,
        depth,
        heads,
        mlp_dim,
        pos=True,
        dim_head=64,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c h w -> b c (h w)"),
            nn.Linear(patch_dim, dim),
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.reverse = nn.Sequential(nn.Linear(dim, patch_dim))
        if pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, ch, dim))
        else:
            self.pos_embedding = None

    def forward(self, x):
        raw_shape = x.shape
        x = self.to_patch_embedding(x)
        x = self.with_pos_embed(x, self.pos_embedding)
        x = self.transformer(x)
        x = self.reverse(x)
        x = x.view(raw_shape).contiguous()
        return x

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


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


class ERNET(SegmentationNetwork):
    def __init__(
        self, num_classes=1, init_ch=3, do_ds=False, conv_op=nn.Conv2d, **kwargs
    ):
        super(ERNET, self).__init__()
        self.params = {"content": None}
        self.conv_op = conv_op
        self.do_ds = do_ds
        self.num_classes = num_classes

        ######## self.model 设置自定义网络 by Sleeep ########
        self.model = ERNet(in_ch_seg=init_ch, out_ch_seg=num_classes,deep_supervison = True)
        ######## self.model 设置自定义网络 by Sleeep ########

        self.name = "ERNET"

    def forward(self, x):
        return self.model(x)
        # if self.do_ds:
        #     return
        #         [self.model(x)]
        # else:
        #     return self.model(x)


"""print layers and params of network"""
if __name__ == "__main__":
    model = ERNET(num_classes=2, init_ch=3)
    dum = torch.rand([1, 3, 256, 256])
    print(model)
    out = model(dum)
    print(out[1].shape)
