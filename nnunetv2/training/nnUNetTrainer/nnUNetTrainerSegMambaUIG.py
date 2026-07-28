import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegMambaUI import (
    CosineAnnealingWarmRestartsWithMultipliers,
    nnUNetTrainerSegMambaUI,
)


class nnUNetTrainerSegMambaUIG(nnUNetTrainerSegMambaUI):
    def configure_optimizers(self):
        aux_prefixes = ("u_decoder", "i_decoder", "u_out", "i_out", "ui_fusion")
        main_params = []
        aux_params = []
        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(aux_prefixes):
                aux_params.append(param)
            else:
                main_params.append(param)

        param_groups = [
            {"params": main_params, "lr": self.initial_lr, "lr_multiplier": 1.0},
            {
                "params": aux_params,
                "lr": self.initial_lr * self.auxiliary_lr_multiplier,
                "lr_multiplier": self.auxiliary_lr_multiplier,
            },
        ]
        optimizer = torch.optim.SGD(
            param_groups,
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingWarmRestartsWithMultipliers(
            optimizer,
            self.initial_lr,
            self.num_epochs,
            first_cycle_steps=self.lr_restart_t0,
            cycle_mult=self.lr_restart_t_mult,
            min_lr=self.lr_min,
        )
        return optimizer, lr_scheduler


class nnUNetTrainerSegMambaUIGStable(nnUNetTrainerSegMambaUIG):
    """Conservative optimizer/loss setup for UIG fusion experiments.

    The U/I auxiliary branches may still need a separate learning-rate policy,
    but the fusion blocks feed directly into the main segmentation branch. They
    therefore get their own parameter group instead of inheriting the aggressive
    auxiliary LR multiplier.
    """

    default_union_loss_weight = 0.2
    default_intersection_loss_weight = 0.1
    default_auxiliary_lr_multiplier = 1.0
    default_fusion_lr_multiplier = 0.5

    def initialize(self):
        self.fusion_lr_multiplier = self.default_fusion_lr_multiplier
        super().initialize()
        self.logger.update_config({"fusion_lr_multiplier": self.fusion_lr_multiplier})

    def configure_optimizers(self):
        aux_prefixes = ("u_decoder", "i_decoder", "u_out", "i_out")
        fusion_prefixes = ("ui_fusion",)
        main_params = []
        aux_params = []
        fusion_params = []

        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(fusion_prefixes):
                fusion_params.append(param)
            elif name.startswith(aux_prefixes):
                aux_params.append(param)
            else:
                main_params.append(param)

        param_groups = [
            {"params": main_params, "lr": self.initial_lr, "lr_multiplier": 1.0},
            {
                "params": aux_params,
                "lr": self.initial_lr * self.auxiliary_lr_multiplier,
                "lr_multiplier": self.auxiliary_lr_multiplier,
            },
            {
                "params": fusion_params,
                "lr": self.initial_lr * self.fusion_lr_multiplier,
                "lr_multiplier": self.fusion_lr_multiplier,
            },
        ]
        optimizer = torch.optim.SGD(
            param_groups,
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingWarmRestartsWithMultipliers(
            optimizer,
            self.initial_lr,
            self.num_epochs,
            first_cycle_steps=self.lr_restart_t0,
            cycle_mult=self.lr_restart_t_mult,
            min_lr=self.lr_min,
        )
        return optimizer, lr_scheduler
