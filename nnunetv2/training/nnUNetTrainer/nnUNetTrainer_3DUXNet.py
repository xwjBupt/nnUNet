"""nnU-Net trainer for the 3D UX-Net architecture from MASILab/3DUX-Net."""

import os

import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2Extention.UXNet3D.network_backbone import UXNET


class nnUNetTrainer_3DUXNet(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.initial_lr = float(os.environ.get("NNUNET_3DUXNET_INITIAL_LR", "1e-4"))
        self.num_epochs = int(os.environ.get("NNUNET_3DUXNET_EPOCHS", "500"))

    def set_deep_supervision_enabled(self, enabled: bool):
        # UXNET exposes one full-resolution segmentation head.
        return None

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        return UXNET(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0.0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.initial_lr, weight_decay=1e-2
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
