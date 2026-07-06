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
