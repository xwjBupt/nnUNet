# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import Optional, Sequence

import torch.nn as nn
import torch 
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    def mamba_forward(self, x):
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(-1, -2).reshape(B, C, *img_dims)

        return x 
    
    def forward(self, x):
        
        x_skip = x

        out_x_1 = self.mamba_forward(x)

        x_2 = rearrange(x, "b c d w h -> b c w d h")

        out_x_2 = self.mamba_forward(x_2)
        
        out_x_2 = rearrange(out_x_2, "b c w d h -> b c d w h")

        x_3 = rearrange(x, "b c d w h -> b c h w d")

        out_x_3 = self.mamba_forward(x_3)
        out_x_3 = rearrange(out_x_3, "b c h w d -> b c d w h")

        out = out_x_1 + out_x_2 + out_x_3

        out = out + x_skip

        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class LargeKernelConv(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.deep_path = nn.Sequential(
            ConvBlock(channels, channels, 5, padding=2),
            ConvBlock(channels, channels, 3, padding=1),
            ConvBlock(channels, channels, 3, padding=2, dilation=2),
        )

        self.shortcut_path = nn.Sequential(
            ConvBlock(channels, channels, 3, padding=1),
            ConvBlock(channels, channels, 1, padding=0)
        )

        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.deep_path(x) + self.shortcut_path(x) + x
        return self.se(out)


class UIPriorFusionBlock(nn.Module):
    """Content-adaptive fusion of U/I priors into the main decoder.

    U is supervised by the adjacent-slice union mask, so it acts as a
    high-recall envelope prior. I is supervised by the adjacent-slice
    intersection mask, so it acts as a high-confidence core prior. Their
    discrepancy describes the uncertain peripheral band. The fusion follows
    an AFF-style local/global attention gate and uses zero-initialized residual
    scales so the pretrained main decoder starts unchanged.
    """

    def __init__(
        self,
        channels: int,
        norm_name: str = "instance",
        reduction: int = 4,
        num_heads: int = 4,
        attention_pool_size: Sequence[int] = (4, 4, 4),
        residual_scale_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        norm = nn.InstanceNorm3d if norm_name == "instance" else nn.BatchNorm3d
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")
        self.attention_pool_size = tuple(attention_pool_size)
        self.residual_scale_limit = residual_scale_limit

        def projection():
            return nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=1, bias=False),
                norm(channels),
                nn.ReLU(inplace=True),
            )

        self.main_projection = projection()
        self.union_projection = projection()
        self.intersection_projection = projection()

        self.union_envelope_gate = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.intersection_core_gate = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.prior_mixer = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False),
            norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm(channels),
        )

        self.local_attention = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )

        self.boundary_refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm(channels),
        )
        self.query_norm = nn.LayerNorm(channels)
        self.prior_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.transformer_ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, channels),
        )
        self.prior_scale = nn.Parameter(torch.zeros(1))
        self.boundary_scale = nn.Parameter(torch.zeros(1))
        self.transformer_scale = nn.Parameter(torch.zeros(1))

    def _scale(self, value: torch.Tensor) -> torch.Tensor:
        if self.residual_scale_limit is None:
            return value
        return float(self.residual_scale_limit) * torch.tanh(value)

    def _pooled_tokens(self, feature: torch.Tensor):
        pooled = F.adaptive_avg_pool3d(feature, self.attention_pool_size)
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        return tokens

    def _tokens_to_feature(self, tokens: torch.Tensor, output_shape: Sequence[int]):
        b, n, c = tokens.shape
        pooled_shape = self.attention_pool_size
        feature = tokens.transpose(1, 2).contiguous().view(b, c, *pooled_shape)
        return F.interpolate(feature, size=output_shape, mode="trilinear", align_corners=False)

    def forward(self, main_feature: torch.Tensor, union_feature: torch.Tensor, intersection_feature: torch.Tensor):
        main_context = self.main_projection(main_feature)
        union_context = self.union_projection(union_feature)
        intersection_context = self.intersection_projection(intersection_feature)

        union_envelope = self.union_envelope_gate(union_context)
        intersection_core = self.intersection_core_gate(intersection_context)
        uncertain_band = union_envelope * (1.0 - intersection_core)

        prior_feature = self.prior_mixer(
            torch.cat(
                (
                    union_context * union_envelope,
                    intersection_context * intersection_core,
                    torch.abs(union_context - intersection_context) * uncertain_band,
                ),
                dim=1,
            )
        )

        fusion_gate = torch.sigmoid(
            self.local_attention(main_context + prior_feature)
            + self.global_attention(main_context + prior_feature)
        )
        adaptive_prior = fusion_gate * prior_feature + (1.0 - fusion_gate) * main_context
        boundary_prior = self.boundary_refine(prior_feature * uncertain_band)
        query_tokens = self.query_norm(self._pooled_tokens(main_context))
        prior_tokens = self.prior_norm(self._pooled_tokens(prior_feature + boundary_prior))
        attended_tokens, _ = self.cross_attention(query_tokens, prior_tokens, prior_tokens, need_weights=False)
        attended_tokens = attended_tokens + self.transformer_ffn(attended_tokens)
        transformer_prior = self._tokens_to_feature(attended_tokens, main_feature.shape[2:])
        return (
            main_feature
            + self._scale(self.prior_scale) * (adaptive_prior - main_context)
            + self._scale(self.boundary_scale) * boundary_prior
            + self._scale(self.transformer_scale) * transformer_prior
        )


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(dims[0], dims[0], kernel_size=3, stride=2, padding=1),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv3d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        cur = 0
        for i in range(4):
            if i < 2:
                gsc = nn.Sequential(
                    *[LargeKernelConv(dims[i]) for j in range(depths[i])]
                )  
                stage = None
                
            else :
                gsc = nn.Sequential(
                    *[LargeKernelConv(dims[i]) for j in range(depths[i])]
                )
                stage = nn.Sequential(
                    *[MambaLayer(dim=dims[i]) for j in range(depths[i])]
                )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i < 2:
                x = self.gscs[i](x)
            else:
                x = self.gscs[i](x)
                x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    """Three-branch SegMamba wrapper compatible with nnU-Net v2.

    The main segmentation branch keeps the original SegMamba encoder/decoder
    topology so pretrained weights remain maximally reusable. The U/I branches
    are auxiliary decoders supervised by three-slice union/intersection masks.
    """

    def __init__(
        self,
        input_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        depths: Sequence[int] = (2, 2, 2, 2),
        feat_size: Sequence[int] = (48, 96, 192, 384),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        hidden_size: int = 768,
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
        deep_supervision: bool = False,
        # Backward-compatible aliases for the original SegMamba constructor.
        in_chans: Optional[int] = None,
        out_chans: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        fusion_levels: Optional[Sequence[str]] = None,
        detach_ui_features: bool = False,
        fusion_residual_scale_limit: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if input_channels is None:
            input_channels = in_chans
        if num_classes is None:
            num_classes = out_chans
        if input_channels is None or num_classes is None:
            raise ValueError(
                "SegMamba requires input_channels and num_classes when used with nnU-Net v2. "
                "You may also pass the old aliases in_chans and out_chans."
            )
        if spatial_dims != 3:
            raise ValueError("This SegMamba implementation is 3D only. Please use it with a 3D nnU-Net configuration.")

        self.hidden_size = hidden_size
        self.in_chans = int(input_channels)
        self.out_chans = int(num_classes)
        self.depths = tuple(depths)
        self.drop_path_rate = drop_path_rate
        self.feat_size = tuple(feat_size)
        self.layer_scale_init_value = layer_scale_init_value
        self.deep_supervision = deep_supervision
        self.fusion_levels = self._normalize_fusion_levels(fusion_levels)
        self.detach_ui_features = bool(detach_ui_features)

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(self.in_chans, 
                                depths=self.depths,
                                dims=self.feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
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
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

        self.u_decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.u_decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.u_decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.u_decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.u_decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.u_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=2)

        self.i_decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.i_decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.i_decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.i_decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.i_decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.i_out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=2)

        fusion_kwargs = {"norm_name": norm_name, "residual_scale_limit": fusion_residual_scale_limit}
        self.ui_fusion_dec3 = UIPriorFusionBlock(self.feat_size[3], **fusion_kwargs)
        self.ui_fusion_dec2 = UIPriorFusionBlock(self.feat_size[2], **fusion_kwargs)
        self.ui_fusion_dec1 = UIPriorFusionBlock(self.feat_size[1], **fusion_kwargs)
        self.ui_fusion_dec0 = UIPriorFusionBlock(self.feat_size[0], **fusion_kwargs)
        self.ui_fusion_out = UIPriorFusionBlock(self.feat_size[0], **fusion_kwargs)

        if self.deep_supervision:
            self.out_dec3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.out_chans)
            self.out_dec2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.out_chans)
            self.out_dec1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.out_chans)
            self.u_out_dec3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=2)
            self.u_out_dec2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=2)
            self.u_out_dec1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=2)
            self.i_out_dec3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=2)
            self.i_out_dec2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=2)
            self.i_out_dec1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=2)
        
        # 在 __init__ 中加载预训练权重
        if pretrained_path is not None:
            print(f"[SegMamba] Loading pretrained weights from {pretrained_path}")

            checkpoint = torch.load(pretrained_path, map_location="cpu")

            # 获取 state_dict
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # 当前模型的 state_dict
            model_dict = self.state_dict()
            filtered_dict = {}

            for k, v in state_dict.items():
                # 去掉 module 前缀
                new_k = k.replace("module.", "")

                # 如果 key 在模型里并且 shape 匹配
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    filtered_dict[new_k] = v
                else:
                    print(f"[SegMamba] Skip {k}: checkpoint {tuple(v.shape)} != model {tuple(model_dict.get(new_k, v).shape)}")

            missing, unexpected = self.load_state_dict(filtered_dict, strict=False)
            print(f"[SegMamba] Loaded: {len(filtered_dict)}, Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    #####
    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    @staticmethod
    def _normalize_fusion_levels(fusion_levels: Optional[Sequence[str]]):
        valid = ("dec3", "dec2", "dec1", "dec0", "out")
        if fusion_levels is None:
            return set(valid)
        if isinstance(fusion_levels, str):
            fusion_levels = [part.strip() for part in fusion_levels.split(",") if part.strip()]
        unknown = set(fusion_levels) - set(valid)
        if unknown:
            raise ValueError(f"Unknown UIG fusion levels {sorted(unknown)}. Valid levels are {valid}.")
        return set(fusion_levels)

    def _maybe_fuse(self, level: str, block: nn.Module, main_feature: torch.Tensor,
                    union_feature: torch.Tensor, intersection_feature: torch.Tensor) -> torch.Tensor:
        if level not in self.fusion_levels:
            return main_feature
        if self.detach_ui_features:
            union_feature = union_feature.detach()
            intersection_feature = intersection_feature.detach()
        return block(main_feature, union_feature, intersection_feature)

    def forward(self, x_in):
        enc1 = self.encoder1(x_in)
        outs = self.vit(enc1)
        
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])

        u_dec3 = self.u_decoder5(enc_hidden, enc4)
        u_dec2 = self.u_decoder4(u_dec3, enc3)
        u_dec1 = self.u_decoder3(u_dec2, enc2)
        u_dec0 = self.u_decoder2(u_dec1, enc1)
        u_out_feat = self.u_decoder1(u_dec0)
        u_seg_out = self.u_out(u_out_feat)

        i_dec3 = self.i_decoder5(enc_hidden, enc4)
        i_dec2 = self.i_decoder4(i_dec3, enc3)
        i_dec1 = self.i_decoder3(i_dec2, enc2)
        i_dec0 = self.i_decoder2(i_dec1, enc1)
        i_out_feat = self.i_decoder1(i_dec0)
        i_seg_out = self.i_out(i_out_feat)

        dec3 = self.decoder5(enc_hidden, enc4)#384*16*16*16
        dec3 = self._maybe_fuse("dec3", self.ui_fusion_dec3, dec3, u_dec3, i_dec3)
        dec2 = self.decoder4(dec3, enc3)#192*32*32*32
        dec2 = self._maybe_fuse("dec2", self.ui_fusion_dec2, dec2, u_dec2, i_dec2)
        dec1 = self.decoder3(dec2, enc2)#96*64*64*64
        dec1 = self._maybe_fuse("dec1", self.ui_fusion_dec1, dec1, u_dec1, i_dec1)
        dec0 = self.decoder2(dec1, enc1)#48*128*128*128
        dec0 = self._maybe_fuse("dec0", self.ui_fusion_dec0, dec0, u_dec0, i_dec0)
        out = self.decoder1(dec0)
        out = self._maybe_fuse("out", self.ui_fusion_out, out, u_out_feat, i_out_feat)
        seg_out = self.out(out)#Class*128*128*128

        if self.deep_supervision:
            return {
                "seg": [seg_out, self.out_dec1(dec1), self.out_dec2(dec2), self.out_dec3(dec3)],
                "u": [u_seg_out, self.u_out_dec1(u_dec1), self.u_out_dec2(u_dec2), self.u_out_dec3(u_dec3)],
                "i": [i_seg_out, self.i_out_dec1(i_dec1), self.i_out_dec2(i_dec2), self.i_out_dec3(i_dec3)],
            }
        elif self.training:
            return {"seg": seg_out, "u": u_seg_out, "i": i_seg_out}
        else:
            return seg_out

    def _copy_matching_module(self, src_prefix, dst_prefix):
        state = self.state_dict()
        copied = {}
        for key, value in state.items():
            if not key.startswith(src_prefix):
                continue
            target_key = dst_prefix + key[len(src_prefix):]
            if target_key in state and state[target_key].shape == value.shape:
                copied[target_key] = value.detach().clone()
        self.load_state_dict(copied, strict=False)
        return len(copied)

    def initialize_auxiliary_branches_from_main(self):
        copied = 0
        copied += self._copy_matching_module("decoder5.", "u_decoder5.")
        copied += self._copy_matching_module("decoder4.", "u_decoder4.")
        copied += self._copy_matching_module("decoder3.", "u_decoder3.")
        copied += self._copy_matching_module("decoder2.", "u_decoder2.")
        copied += self._copy_matching_module("decoder1.", "u_decoder1.")
        copied += self._copy_matching_module("decoder5.", "i_decoder5.")
        copied += self._copy_matching_module("decoder4.", "i_decoder4.")
        copied += self._copy_matching_module("decoder3.", "i_decoder3.")
        copied += self._copy_matching_module("decoder2.", "i_decoder2.")
        copied += self._copy_matching_module("decoder1.", "i_decoder1.")
        out_state = {}
        state = self.state_dict()
        for head in ("u_out", "i_out"):
            for suffix in ("conv.conv.weight", "conv.conv.bias"):
                src = f"out.{suffix}"
                dst = f"{head}.{suffix}"
                if src in state and dst in state:
                    src_value = state[src]
                    dst_value = state[dst]
                    if src_value.shape == dst_value.shape:
                        out_state[dst] = src_value.detach().clone()
                    elif src_value.shape[0] >= dst_value.shape[0] and src_value.shape[1:] == dst_value.shape[1:]:
                        out_state[dst] = src_value[:dst_value.shape[0]].detach().clone()
        self.load_state_dict(out_state, strict=False)
        copied += len(out_state)
        print(f"[SegMambaUI] Initialized auxiliary U/I branches from main branch: {copied} tensors copied.")

    def load_from(self, pretrained_path):
            if pretrained_path is not None:
                print(f"[SegMamba 手术式加载] 正在从 {pretrained_path} 载入并改造预训练权重...")
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "network_weights" in checkpoint:
                    state_dict = checkpoint["network_weights"]
                else:
                    state_dict = checkpoint

                model_dict = self.state_dict()
                expected_dict = {}

                for k, v in state_dict.items():
                    # 移除 DDP 包装产生的 module. 前缀
                    key = k[7:] if k.startswith("module.") else k
                    
                    if key in model_dict:
                        # 💥 核心手术 1: 修复输入层通道不匹配 (Checkpoint 为 4，Model 为 1)
                        if key == "encoder1.layer.conv1.conv.weight" and v.shape != model_dict[key].shape:
                            print(f" -> 改造输入层通道: {v.shape} -> {model_dict[key].shape}")
                            # 截取第一通道，或者取 4 个通道的平均值均可，这里采用取平均
                            v = v_modified = torch.mean(v, dim=1, keepdim=True)
                            
                        if key == "encoder1.layer.conv3.conv.weight" and v.shape != model_dict[key].shape:
                            print(f" -> 改造输入辅助层通道: {v.shape} -> {model_dict[key].shape}")
                            v = torch.mean(v, dim=1, keepdim=True)

                        # 💥 核心手术 2: 修复输出层类别数不匹配 (Checkpoint 为 4，Model 为 2)
                        if key == "out.conv.conv.weight" and v.shape != model_dict[key].shape:
                            print(f" -> 截取输出层类别权重: {v.shape} -> {model_dict[key].shape}")
                            v = v[:model_dict[key].shape[0], ...] # 截取前 2 个分类
                            
                        if key == "out.conv.conv.bias" and v.shape != model_dict[key].shape:
                            print(f" -> 截取输出层偏置: {v.shape} -> {model_dict[key].shape}")
                            v = v[:model_dict[key].shape[0]]

                        # 检查形状是否完美对齐
                        if v.shape == model_dict[key].shape:
                            expected_dict[key] = v
                        else:
                            print(f" -> 警告: 形状依然不匹配，跳过 {key} {v.shape} vs {model_dict[key].shape}")

                print(f"[SegMamba] 成功手术对齐并加载了 {len(expected_dict)} 个权重项！")
                self.load_state_dict(expected_dict, strict=False)
                self.initialize_auxiliary_branches_from_main()
if __name__ == '__main__':
    
    SegMamba = SegMamba(input_channels = 1,num_classes = 1).cuda()
    dummy = torch.randn(1,1,128,128,128).cuda()
    out = SegMamba(dummy)
    print (out.shape)
    
