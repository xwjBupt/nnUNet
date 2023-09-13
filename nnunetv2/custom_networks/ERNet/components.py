import torch
import torch.nn as nn
import torch.nn.functional as F

# from inplace_abn import ABN
import functools
from collections import OrderedDict

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

BN_MOMENTUM = 0.01


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


class VAEBasicUp(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        deconv: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(VAEBasicUp, self).__init__()
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

        self.conv1 = conv3x3(out_ch, out_ch)
        self.bn1 = self._norm_layer(out_ch)

        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = self._norm_layer(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        if self.deconv:
            x = self.fuse(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.fuse(x)

        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.activation(x)

        return x


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, use_syncbn=False):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            syncbn(out_ch, momentum=BN_MOMENTUM, activation="identity")
            if use_syncbn
            else nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            syncbn(out_ch, momentum=BN_MOMENTUM, activation="identity")
            if use_syncbn
            else nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ASPP(nn.Module):
    def __init__(
        self,
        in_ch,
        reduction=5,
        dilations=[6, 12, 18],
        norm_layer=None,
        activation=None,
    ):
        super(ASPP, self).__init__()
        middle = in_ch // reduction
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        else:
            self._norm_layer = norm_layer

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation

        self.conv_1x1_1 = nn.Conv2d(in_ch, middle, kernel_size=1)
        self.bn_conv_1x1_1 = self._norm_layer(middle)

        self.conv_3x3_1 = nn.Conv2d(
            in_ch,
            middle,
            kernel_size=3,
            stride=1,
            padding=dilations[0],
            dilation=dilations[0],
        )
        self.bn_conv_3x3_1 = self._norm_layer(middle)

        self.conv_3x3_2 = nn.Conv2d(
            in_ch,
            middle,
            kernel_size=3,
            stride=1,
            padding=dilations[1],
            dilation=dilations[1],
        )
        self.bn_conv_3x3_2 = self._norm_layer(middle)

        self.conv_3x3_3 = nn.Conv2d(
            in_ch,
            middle,
            kernel_size=3,
            stride=1,
            padding=dilations[2],
            dilation=dilations[2],
        )
        self.bn_conv_3x3_3 = self._norm_layer(middle)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_ch, middle, kernel_size=1)
        self.bn_conv_1x1_2 = self._norm_layer(middle)

        self.conv_1x1_3 = nn.Conv2d(
            reduction * middle, in_ch, kernel_size=1
        )  # (1280 = 5*256)
        self.bn_conv_1x1_3 = self._norm_layer(in_ch)

        self.conv_3x3_0 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.bn_conv_3x3_0 = self._norm_layer(in_ch)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        out_1x1 = self.activation(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = self.activation(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = self.activation(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = self.activation(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = self.activation(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(
            out_img, size=(feature_map_h, feature_map_w), mode="bilinear"
        )

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = self.activation(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        out = self.activation(self.bn_conv_3x3_0(self.conv_3x3_0(out)))
        return out


class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes):
        super(ASPP_Bottleneck, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(4 * 512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(
            4 * 512, 256, kernel_size=3, stride=1, padding=6, dilation=6
        )
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(
            4 * 512, 256, kernel_size=3, stride=1, padding=12, dilation=12
        )
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(
            4 * 512, 256, kernel_size=3, stride=1, padding=18, dilation=18
        )
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(4 * 512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 4*512, h/16, w/16))

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        # (shape: (batch_size, 256, h/16, w/16))
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        # (shape: (batch_size, 512, 1, 1))
        out_img = self.avg_pool(feature_map)
        # (shape: (batch_size, 256, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        # (shape: (batch_size, 256, h/16, w/16))
        out_img = F.upsample(
            out_img, size=(feature_map_h, feature_map_w), mode="bilinear"
        )

        # (shape: (batch_size, 1280, h/16, w/16))
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        # (shape: (batch_size, 256, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        # (shape: (batch_size, num_classes, h/16, w/16))
        out = self.conv_1x1_4(out)

        return out


class PPM(nn.Module):
    """Pyramid pooling module"""

    def __init__(
        self,
        inc,
        outc,
        middle=4,
        norm_layer=None,
        pool_size=[1, 2, 3, 5],
        activation=None,
        **kwargs
    ):
        super(PPM, self).__init__()
        inter_channels = int(inc / middle)  # 这里N=4与原文一致
        self.pool_size = pool_size
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = norm_layer

        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation

        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv1x1(inc, inter_channels)),
                    ("bn1", self.norm_layer(inter_channels)),
                    (
                        "activation",
                        self.activation,
                    ),
                    ("avg", nn.AdaptiveAvgPool2d(self.pool_size[0])),
                ]
            )
        )
        self.conv2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv1x1(inc, inter_channels)),
                    ("bn1", self.norm_layer(inter_channels)),
                    (
                        "activation",
                        self.activation,
                    ),
                    ("avg", nn.AdaptiveAvgPool2d(self.pool_size[1])),
                ]
            )
        )
        self.conv3 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv1x1(inc, inter_channels)),
                    ("bn1", self.norm_layer(inter_channels)),
                    (
                        "activation",
                        self.activation,
                    ),
                    ("avg", nn.AdaptiveAvgPool2d(self.pool_size[2])),
                ]
            )
        )
        self.conv4 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv1x1(inc, inter_channels)),
                    ("bn1", self.norm_layer(inter_channels)),
                    (
                        "activation",
                        self.activation,
                    ),
                    ("avg", nn.AdaptiveAvgPool2d(self.pool_size[3])),
                ]
            )
        )
        self.fuse = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv1x1(4 * inter_channels, outc)),
                    ("bn1", self.norm_layer(outc)),
                    (
                        "activation",
                        self.activation,
                    ),
                ]
            )
        )

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(x), size)
        feat2 = self.upsample(self.conv2(x), size)
        feat3 = self.upsample(self.conv3(x), size)
        feat4 = self.upsample(self.conv4(x), size)
        x = torch.cat([feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        x = self.fuse(x)
        return x


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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
