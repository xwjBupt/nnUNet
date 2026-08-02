import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegMambaUI import (
    CosineAnnealingWarmRestartsWithMultipliers,
    nnUNetTrainerSegMambaUI,
)
from nnunetv2.utilities.collate_outputs import collate_outputs


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


class nnUNetTrainerSegMambaUIGStableHierarchy(nnUNetTrainerSegMambaUIGStable):
    """Add soft I <= Seg <= U consistency supervision to the Stable trainer."""

    default_hierarchy_loss_weight = 0.05

    def initialize(self):
        self.hierarchy_loss_weight = self.default_hierarchy_loss_weight
        super().initialize()
        self.logger.update_config({"hierarchy_loss_weight": self.hierarchy_loss_weight})

    @staticmethod
    def _foreground_probability(logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)[:, 1:].sum(dim=1, keepdim=True)

    def _compute_additional_branch_loss(self, seg_output, u_output, i_output):
        if u_output is None or i_output is None:
            return super()._compute_additional_branch_loss(seg_output, u_output, i_output)

        seg_outputs = self._as_list(seg_output)
        union_outputs = self._as_list(u_output)
        intersection_outputs = self._as_list(i_output)
        if not (len(seg_outputs) == len(union_outputs) == len(intersection_outputs)):
            raise RuntimeError(
                "Hierarchy loss requires matching Seg/U/I deep-supervision outputs, got "
                f"{len(seg_outputs)}/{len(union_outputs)}/{len(intersection_outputs)}."
            )

        weights = np.array([1 / (2 ** index) for index in range(len(seg_outputs))], dtype=np.float32)
        if len(weights) > 1:
            weights[-1] = 1e-6 if self.is_ddp else 0.0
        weights /= weights.sum()

        hierarchy_loss = seg_outputs[0].new_zeros((), dtype=torch.float32)
        for weight, seg_logits, union_logits, intersection_logits in zip(
            weights, seg_outputs, union_outputs, intersection_outputs
        ):
            if weight == 0:
                continue
            seg_probability = self._foreground_probability(seg_logits)
            union_probability = self._foreground_probability(union_logits)
            intersection_probability = self._foreground_probability(intersection_logits)

            violation = (
                F.relu(intersection_probability - seg_probability)
                + F.relu(seg_probability - union_probability)
                + 0.5 * F.relu(intersection_probability - union_probability)
            )
            relevance = torch.maximum(
                torch.maximum(seg_probability, union_probability), intersection_probability
            ).detach()
            normalized_violation = (violation * relevance).sum() / relevance.sum().clamp_min(1e-6)
            hierarchy_loss = hierarchy_loss + float(weight) * normalized_violation

        return self.hierarchy_loss_weight * hierarchy_loss

    def _ensure_aux_logger_keys(self):
        super()._ensure_aux_logger_keys()
        local_logging = self.logger.local_logger.my_fantastic_logging
        local_logging.setdefault("train_losses_hierarchy", [])
        local_logging.setdefault("val_losses_hierarchy", [])

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        if "loss_additional" in outputs:
            self.logger.log(
                "train_losses_hierarchy",
                float(np.mean(outputs["loss_additional"])),
                self.current_epoch,
            )

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)
        if "loss_additional" in outputs:
            self.logger.log(
                "val_losses_hierarchy",
                float(np.mean(outputs["loss_additional"])),
                self.current_epoch,
            )


class nnUNetTrainerSegMambaUIGStableOffset(nnUNetTrainerSegMambaUIGStable):
    """Build U/I targets from three D-axis slices separated by a fixed offset."""

    ui_slice_offset_voxels = 1

    def initialize(self):
        super().initialize()
        self.logger.update_config({"ui_slice_offset_voxels": self.ui_slice_offset_voxels})

    def _make_union_intersection_targets(self, target):
        target_list = self._as_list(target)
        base_target = target_list[0]
        foreground = (base_target > 0).float()
        if foreground.ndim != 5:
            raise RuntimeError(f"U/I supervision expects Bx1xDxHxW targets, got {tuple(foreground.shape)}")

        offset = int(self.ui_slice_offset_voxels)
        depth = foreground.shape[2]
        padded = F.pad(foreground, (0, 0, 0, 0, offset, offset), mode="replicate")
        previous_slice = padded[:, :, :depth]
        center_slice = padded[:, :, offset:offset + depth]
        next_slice = padded[:, :, 2 * offset:2 * offset + depth]
        stacked = torch.stack((previous_slice, center_slice, next_slice), dim=0)
        union_full = stacked.amax(dim=0)
        intersection_full = stacked.amin(dim=0)

        union_targets = []
        intersection_targets = []
        for deep_supervision_target in target_list:
            shape = deep_supervision_target.shape[2:]
            if tuple(shape) == tuple(union_full.shape[2:]):
                union_targets.append(union_full.long())
                intersection_targets.append(intersection_full.long())
            else:
                union_targets.append(F.interpolate(union_full, size=shape, mode="nearest").long())
                intersection_targets.append(F.interpolate(intersection_full, size=shape, mode="nearest").long())
        if isinstance(target, (list, tuple)):
            return union_targets, intersection_targets
        return union_targets[0], intersection_targets[0]


class nnUNetTrainerSegMambaUIGStableSliceOffset3(nnUNetTrainerSegMambaUIGStableOffset):
    """Use U/I targets from slices at -3, 0 and +3 voxels on the D axis."""

    ui_slice_offset_voxels = 3


class nnUNetTrainerSegMambaUIGStableSliceOffset5(nnUNetTrainerSegMambaUIGStableOffset):
    """Use U/I targets from slices at -5, 0 and +5 voxels on the D axis."""

    ui_slice_offset_voxels = 5
