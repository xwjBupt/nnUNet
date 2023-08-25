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

# from custom_networks.UXNet_3D.network_backbone import UXNET
# from custom_networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.networks.nets import UNETR, SwinUNETR
from nnunetv2.custom_networks.nnformer.nnFormer_seg import nnFormer


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
    return model
