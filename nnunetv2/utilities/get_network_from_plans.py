from dynamic_network_architectures.architectures.unet import (
    PlainConvUNet,
    ResidualEncoderUNet,
)
from dynamic_network_architectures.building_blocks.helper import (
    get_matching_instancenorm,
    convert_dim_to_conv_op,
)
from dynamic_network_architectures.initialization.weight_init import (
    init_last_bn_before_add_to_0,
)
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from torch import nn
import torch

# from custom_networks.UXNet_3D.network_backbone import UXNET
# from custom_networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.networks.nets import UNETR, SwinUNETR
from nnunetv2.custom_networks.nnformer.nnFormer_seg import nnFormer
from nnunetv2.custom_networks.UXNet_3D.network_backbone import UXNET
from nnunetv2.custom_networks.DMFNet import DMFNet
from nnunetv2.custom_networks.PHTrans.phtrans import PHTrans
from nnunetv2.custom_networks.ERNet.ERNet import ERNet


def get_network_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        "PlainConvUNet": PlainConvUNet,
        "ResidualEncoderUNet": ResidualEncoderUNet,
    }
    kwargs = {
        "PlainConvUNet": {
            "conv_bias": True,
            "norm_op": get_matching_instancenorm(conv_op),
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        },
        "ResidualEncoderUNet": {
            "conv_bias": True,
            "norm_op": get_matching_instancenorm(conv_op),
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        },
    }
    assert segmentation_network_class_name in mapping.keys(), (
        "The network architecture specified by the plans file "
        "is non-standard (maybe your own?). Yo'll have to dive "
        "into either this "
        "function (get_network_from_plans) or "
        "the init of your nnUNetModule to accomodate that."
    )
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        "n_conv_per_stage"
        if network_class != ResidualEncoderUNet
        else "n_blocks_per_stage": configuration_manager.n_conv_per_stage_encoder,
        "n_conv_per_stage_decoder": configuration_manager.n_conv_per_stage_decoder,
    }
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[
            min(
                configuration_manager.UNet_base_num_features * 2**i,
                configuration_manager.unet_max_num_features,
            )
            for i in range(num_stages)
        ],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model


def get_custom_network_from_plans(
    segmentation_network_class_name,
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True,
):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)
    mapping = {
        # "3DUXNET": UXNET,
        # "TransBTS": TransBTS,
        "SwinUNETR": SwinUNETR,
        "nnFormer": nnFormer,
        "UNETR": UNETR,
        "UXNET": UXNET,
        "DMFNet": DMFNet,
        "PHTrans": PHTrans,
        "ERNet": ERNet,
    }
    # kwargs = {
    #     "PlainConvUNet": {
    #         "conv_bias": True,
    #         "norm_op": get_matching_instancenorm(conv_op),
    #         "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    #         "dropout_op": None,
    #         "dropout_op_kwargs": None,
    #         "nonlin": nn.LeakyReLU,
    #         "nonlin_kwargs": {"inplace": True},
    #     },
    #     "ResidualEncoderUNet": {
    #         "conv_bias": True,
    #         "norm_op": get_matching_instancenorm(conv_op),
    #         "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    #         "dropout_op": None,
    #         "dropout_op_kwargs": None,
    #         "nonlin": nn.LeakyReLU,
    #         "nonlin_kwargs": {"inplace": True},
    #     },
    # }
    assert segmentation_network_class_name in mapping.keys(), (
        "The network architecture specified by the plans file "
        "is non-standard (maybe your own?). Yo'll have to dive "
        "into either this "
        "function (get_network_from_plans) or "
        "the init of your nnUNetModule to accomodate that. Right now only support {}".format(
            mapping.keys()
        )
    )
    # network_class = mapping[segmentation_network_class_name]

    # conv_or_blocks_per_stage = {
    #     "n_conv_per_stage"
    #     if network_class != ResidualEncoderUNet
    #     else "n_blocks_per_stage": configuration_manager.n_conv_per_stage_encoder,
    #     "n_conv_per_stage_decoder": configuration_manager.n_conv_per_stage_decoder,
    # }
    # network class name!!
    # model = network_class(
    #     input_channels=num_input_channels,
    #     n_stages=num_stages,
    #     features_per_stage=[
    #         min(
    #             configuration_manager.UNet_base_num_features * 2**i,
    #             configuration_manager.unet_max_num_features,
    #         )
    #         for i in range(num_stages)
    #     ],
    #     conv_op=conv_op,
    #     kernel_sizes=configuration_manager.conv_kernel_sizes,
    #     strides=configuration_manager.pool_op_kernel_sizes,
    #     num_classes=label_manager.num_segmentation_heads,
    #     deep_supervision=deep_supervision,
    #     **conv_or_blocks_per_stage,
    #     **kwargs[segmentation_network_class_name]
    # )
    out_classes = label_manager.num_segmentation_heads
    input_channels = num_input_channels
    pretrain_classes = 1
    pretrain = False
    if segmentation_network_class_name == "nnFormer":
        if pretrain:
            from networks.nnFormer.nnFormer_seg import final_patch_expanding

            final_layer = []
            model = nnFormer(
                input_channels=input_channels, num_classes=pretrain_classes
            )
            model.load_state_dict(pretrain)
            final_layer.append(
                final_patch_expanding(192, out_classes, patch_size=[2, 4, 4])
            )
            model.final = nn.ModuleList(final_layer)

        else:
            model = nnFormer(
                crop_size=configuration_manager.patch_size,
                input_channels=input_channels,
                num_classes=out_classes,
                conv_op=conv_op,
                deep_supervision=deep_supervision,
            )
            model.apply(InitWeights_He(1e-2))
    if segmentation_network_class_name == "SwinUNETR":
        if pretrain:
            pass
        else:
            model = SwinUNETR(
                img_size=configuration_manager.patch_size,
                in_channels=input_channels,
                out_channels=out_classes,
                feature_size=48,
                use_checkpoint=False,
            )
            model.apply(InitWeights_He(1e-2))
    if segmentation_network_class_name == "UXNET":
        if pretrain:
            pass
        else:
            model = UXNET(
                in_chans=input_channels,
                out_chans=out_classes,
                depths=[2, 2, 2, 2],
                feat_size=[48, 96, 192, 384],
                drop_path_rate=0,
                layer_scale_init_value=1e-6,
                spatial_dims=3,
            )
            model.apply(InitWeights_He(1e-2))
            # some times use the command: rm -rf ~/.nv
    if segmentation_network_class_name == "DMFNet":
        if pretrain:
            pass
        else:
            model = DMFNet(
                c=input_channels,
                num_classes=out_classes,
                deep_supervision=deep_supervision,
            )
            model.apply(InitWeights_He(1e-2))
            # some times use the command: rm -rf ~/.nv
    if segmentation_network_class_name == "PHTrans":
        num_pool = len(configuration_manager.pool_op_kernel_sizes) - 1
        num_only_conv_stage = 3
        if pretrain:
            pass
        else:
            model = PHTrans(
                img_size=configuration_manager.patch_size,
                base_num_features=configuration_manager.UNet_base_num_features,
                num_classes=out_classes,
                num_pool=num_pool,
                image_channels=input_channels,
                deep_supervision=deep_supervision,
                max_num_features=configuration_manager.unet_max_num_features,
                depths=[2 for i in range(num_pool - num_only_conv_stage + 1)],
                num_only_conv_stage=num_only_conv_stage,
                num_heads=[4, 16, 8, 8],  # len(num_heads) = len(depths)
                window_size=[4, 5, 5],
                pool_op_kernel_sizes=configuration_manager.pool_op_kernel_sizes,
                conv_kernel_sizes=configuration_manager.conv_kernel_sizes,
                dropout_p=0.0,
                drop_path_rate=0.2,
            )
            # model.apply(InitWeights_He(1e-2))
            # some times use the command: rm -rf ~/.nv
    if segmentation_network_class_name == "ERNet":
        if pretrain:
            pass
        else:
            model = ERNet(
                in_ch_seg=input_channels,
                out_ch_seg=out_classes,
                deep_supervision=deep_supervision,
                trans_mode="baseformerV2",
                depth=2,
                heads=16,
                dropout=0,
                mlp_dim=2048,
            )
            model.apply(InitWeights_He(1e-2))
    return model


if __name__ == "__main__":
    dummy = torch.rand([2, 3, 256, 256])
    model = ERNet(
        in_ch_seg=3,
        out_ch_seg=2,
        deep_supervision=True,
        trans_mode="baseformerV2",
        depth=1,
        heads=16,
        dropout=0.1,
        mlp_dim=2048,
    )
    print(model)
    # model = SwinUNETR(
    #     img_size=[16, 320, 320],
    #     in_channels=1,
    #     out_channels=2,
    #     feature_size=48,
    #     use_checkpoint=False,
    # )
    out = model(dummy)
    for i in out:
        print(i.shape)
