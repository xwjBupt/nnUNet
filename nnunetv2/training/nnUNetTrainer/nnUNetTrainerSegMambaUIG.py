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

    def _compute_additional_branch_loss(
        self, seg_output, u_output, i_output, u_target=None, i_target=None
    ):
        if u_output is None or i_output is None:
            return super()._compute_additional_branch_loss(
                seg_output, u_output, i_output, u_target=u_target, i_target=i_target
            )

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


class nnUNetTrainerSegMambaUIGStableHierarchyConservative(
    nnUNetTrainerSegMambaUIGStableHierarchy
):
    """Use detached, margin-aware and asymmetric Seg/U/I containment.

    The intersection branch remains a strong lower-bound cue for missed
    hemorrhage. The union branch is a weaker upper-bound cue because an
    underestimated union can otherwise suppress true-positive segmentation.
    """

    default_hierarchy_margin = 0.05
    default_hierarchy_lower_weight = 1.0
    default_hierarchy_upper_weight = 0.25
    default_hierarchy_order_weight = 0.25

    def initialize(self):
        self.hierarchy_margin = self.default_hierarchy_margin
        self.hierarchy_lower_weight = self.default_hierarchy_lower_weight
        self.hierarchy_upper_weight = self.default_hierarchy_upper_weight
        self.hierarchy_order_weight = self.default_hierarchy_order_weight
        super().initialize()
        self.logger.update_config(
            {
                "hierarchy_guidance_detached": True,
                "hierarchy_margin": self.hierarchy_margin,
                "hierarchy_lower_weight": self.hierarchy_lower_weight,
                "hierarchy_upper_weight": self.hierarchy_upper_weight,
                "hierarchy_order_weight": self.hierarchy_order_weight,
            }
        )
        self.print_to_log_file(
            "Conservative hierarchy: "
            f"weight={self.hierarchy_loss_weight}, margin={self.hierarchy_margin}, "
            f"lower/upper/order={self.hierarchy_lower_weight}/"
            f"{self.hierarchy_upper_weight}/{self.hierarchy_order_weight}, "
            "U/I guidance detached from the main containment term."
        )

    def _compute_additional_branch_loss(
        self, seg_output, u_output, i_output, u_target=None, i_target=None
    ):
        if u_output is None or i_output is None:
            return nnUNetTrainerSegMambaUIGStable._compute_additional_branch_loss(
                self,
                seg_output,
                u_output,
                i_output,
                u_target=u_target,
                i_target=i_target,
            )

        seg_outputs = self._as_list(seg_output)
        union_outputs = self._as_list(u_output)
        intersection_outputs = self._as_list(i_output)
        if not (len(seg_outputs) == len(union_outputs) == len(intersection_outputs)):
            raise RuntimeError(
                "Conservative hierarchy loss requires matching Seg/U/I outputs, got "
                f"{len(seg_outputs)}/{len(union_outputs)}/{len(intersection_outputs)}."
            )

        weights = np.array(
            [1 / (2 ** index) for index in range(len(seg_outputs))], dtype=np.float32
        )
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

            # U/I supervise their own targets. For main-branch containment they
            # are fixed guidance, so the loss cannot be reduced by moving U/I
            # toward an erroneous main prediction.
            union_guidance = union_probability.detach()
            intersection_guidance = intersection_probability.detach()
            margin = float(self.hierarchy_margin)

            lower_violation = F.relu(
                intersection_guidance - seg_probability - margin
            )
            upper_violation = F.relu(
                seg_probability - union_guidance - margin
            )
            order_violation = F.relu(
                intersection_probability - union_probability - margin
            )
            violation = (
                self.hierarchy_lower_weight * lower_violation
                + self.hierarchy_upper_weight * upper_violation
                + self.hierarchy_order_weight * order_violation
            )

            relevance = torch.maximum(
                torch.maximum(seg_probability, union_guidance), intersection_guidance
            ).detach()
            normalized_violation = (
                (violation * relevance).sum() / relevance.sum().clamp_min(1e-6)
            )
            hierarchy_loss = hierarchy_loss + float(weight) * normalized_violation

        return self.hierarchy_loss_weight * hierarchy_loss


class nnUNetTrainerSegMambaUIGStableHierarchyHighResReliable(
    nnUNetTrainerSegMambaUIGStableHierarchy
):
    """Apply target-reliability-weighted hierarchy at the two finest scales."""

    default_hierarchy_reliability_floor = 0.5
    default_hierarchy_active_scales = 2

    def initialize(self):
        self.hierarchy_reliability_floor = self.default_hierarchy_reliability_floor
        self.hierarchy_active_scales = self.default_hierarchy_active_scales
        super().initialize()
        self.logger.update_config(
            {
                "hierarchy_variant": "highres_target_reliable",
                "hierarchy_active_scales": self.hierarchy_active_scales,
                "hierarchy_scale_weights": [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
                "hierarchy_reliability_floor": self.hierarchy_reliability_floor,
                "hierarchy_margin": 0.0,
                "hierarchy_guidance_detached": False,
                "hierarchy_lower_weight": 1.0,
                "hierarchy_upper_weight": 1.0,
                "hierarchy_order_weight": 0.5,
            }
        )
        self.print_to_log_file(
            "HighRes Target-Reliable hierarchy: "
            f"weight={self.hierarchy_loss_weight}, active_scales="
            f"{self.hierarchy_active_scales}, reliability_floor="
            f"{self.hierarchy_reliability_floor}, margin=0, "
            "lower/upper/order=1/1/0.5."
        )

    def _compute_additional_branch_loss(
        self, seg_output, u_output, i_output, u_target=None, i_target=None
    ):
        if u_output is None or i_output is None:
            return nnUNetTrainerSegMambaUIGStable._compute_additional_branch_loss(
                self,
                seg_output,
                u_output,
                i_output,
                u_target=u_target,
                i_target=i_target,
            )
        if u_target is None or i_target is None:
            raise RuntimeError(
                "HighRes Target-Reliable hierarchy requires U/I supervision targets."
            )

        seg_outputs = self._as_list(seg_output)
        union_outputs = self._as_list(u_output)
        intersection_outputs = self._as_list(i_output)
        union_targets = self._as_list(u_target)
        intersection_targets = self._as_list(i_target)
        output_counts = {
            len(seg_outputs),
            len(union_outputs),
            len(intersection_outputs),
            len(union_targets),
            len(intersection_targets),
        }
        if len(output_counts) != 1:
            raise RuntimeError(
                "HighRes Target-Reliable hierarchy requires matching Seg/U/I outputs "
                "and targets, got "
                f"{len(seg_outputs)}/{len(union_outputs)}/{len(intersection_outputs)}/"
                f"{len(union_targets)}/{len(intersection_targets)}."
            )

        weights = np.zeros(len(seg_outputs), dtype=np.float32)
        active_scales = min(self.hierarchy_active_scales, len(weights))
        weights[:active_scales] = np.asarray(
            [1 / (2 ** index) for index in range(active_scales)], dtype=np.float32
        )
        weights /= weights.sum()

        hierarchy_loss = seg_outputs[0].new_zeros((), dtype=torch.float32)
        reliability_floor = float(self.hierarchy_reliability_floor)
        reliability_range = 1.0 - reliability_floor
        for weight, seg_logits, union_logits, intersection_logits, union_label, intersection_label in zip(
            weights,
            seg_outputs,
            union_outputs,
            intersection_outputs,
            union_targets,
            intersection_targets,
        ):
            if weight == 0:
                continue

            seg_probability = self._foreground_probability(seg_logits)
            union_probability = self._foreground_probability(union_logits)
            intersection_probability = self._foreground_probability(intersection_logits)
            union_label = (union_label > 0).float()
            intersection_label = (intersection_label > 0).float()

            union_reliability = reliability_floor + reliability_range * (
                1.0 - torch.abs(union_probability.detach() - union_label)
            )
            intersection_reliability = reliability_floor + reliability_range * (
                1.0 - torch.abs(intersection_probability.detach() - intersection_label)
            )

            violation = (
                intersection_reliability
                * F.relu(intersection_probability - seg_probability)
                + union_reliability * F.relu(seg_probability - union_probability)
                + 0.5 * F.relu(intersection_probability - union_probability)
            )
            relevance = torch.maximum(
                torch.maximum(seg_probability, union_probability), intersection_probability
            ).detach()
            normalized_violation = (
                (violation * relevance).sum() / relevance.sum().clamp_min(1e-6)
            )
            hierarchy_loss = hierarchy_loss + float(weight) * normalized_violation

        return self.hierarchy_loss_weight * hierarchy_loss


class nnUNetTrainerSegMambaUIGStableHierarchyFusionAlignedReliable(
    nnUNetTrainerSegMambaUIGStableHierarchyHighResReliable
):
    """Keep reliable high-resolution constraints and anchor the dec2 fusion scale."""

    default_hierarchy_active_scales = 3
    default_hierarchy_reliable_scales = 2

    def initialize(self):
        self.hierarchy_reliable_scales = self.default_hierarchy_reliable_scales
        super().initialize()
        self.logger.update_config(
            {
                "hierarchy_variant": "fusion_aligned_reliable",
                "hierarchy_active_scales": self.hierarchy_active_scales,
                "hierarchy_reliable_scales": self.hierarchy_reliable_scales,
                "hierarchy_scale_weights": [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 0.0],
                "hierarchy_fusion_anchor_scale": "dec2",
                "hierarchy_fusion_anchor_reliability_weighted": False,
            }
        )
        self.print_to_log_file(
            "Fusion-Aligned Reliable hierarchy: "
            f"weight={self.hierarchy_loss_weight}, active_scales="
            f"{self.hierarchy_active_scales}, reliable_scales="
            f"{self.hierarchy_reliable_scales}, weights=4/7,2/7,1/7,0; "
            "dec2 uses the original unattenuated hierarchy constraint."
        )

    def _compute_additional_branch_loss(
        self, seg_output, u_output, i_output, u_target=None, i_target=None
    ):
        if u_output is None or i_output is None:
            return nnUNetTrainerSegMambaUIGStable._compute_additional_branch_loss(
                self,
                seg_output,
                u_output,
                i_output,
                u_target=u_target,
                i_target=i_target,
            )
        if u_target is None or i_target is None:
            raise RuntimeError(
                "Fusion-Aligned Reliable hierarchy requires U/I supervision targets."
            )

        seg_outputs = self._as_list(seg_output)
        union_outputs = self._as_list(u_output)
        intersection_outputs = self._as_list(i_output)
        union_targets = self._as_list(u_target)
        intersection_targets = self._as_list(i_target)
        output_counts = {
            len(seg_outputs),
            len(union_outputs),
            len(intersection_outputs),
            len(union_targets),
            len(intersection_targets),
        }
        if len(output_counts) != 1:
            raise RuntimeError(
                "Fusion-Aligned Reliable hierarchy requires matching Seg/U/I outputs "
                "and targets, got "
                f"{len(seg_outputs)}/{len(union_outputs)}/{len(intersection_outputs)}/"
                f"{len(union_targets)}/{len(intersection_targets)}."
            )

        weights = np.zeros(len(seg_outputs), dtype=np.float32)
        active_scales = min(self.hierarchy_active_scales, len(weights))
        weights[:active_scales] = np.asarray(
            [1 / (2 ** index) for index in range(active_scales)], dtype=np.float32
        )
        weights /= weights.sum()

        hierarchy_loss = seg_outputs[0].new_zeros((), dtype=torch.float32)
        reliability_floor = float(self.hierarchy_reliability_floor)
        reliability_range = 1.0 - reliability_floor
        for scale_index, (
            weight,
            seg_logits,
            union_logits,
            intersection_logits,
            union_label,
            intersection_label,
        ) in enumerate(
            zip(
                weights,
                seg_outputs,
                union_outputs,
                intersection_outputs,
                union_targets,
                intersection_targets,
            )
        ):
            if weight == 0:
                continue

            seg_probability = self._foreground_probability(seg_logits)
            union_probability = self._foreground_probability(union_logits)
            intersection_probability = self._foreground_probability(intersection_logits)

            if scale_index < self.hierarchy_reliable_scales:
                union_label = (union_label > 0).float()
                intersection_label = (intersection_label > 0).float()
                union_reliability = reliability_floor + reliability_range * (
                    1.0 - torch.abs(union_probability.detach() - union_label)
                )
                intersection_reliability = reliability_floor + reliability_range * (
                    1.0
                    - torch.abs(
                        intersection_probability.detach() - intersection_label
                    )
                )
            else:
                # dec2 is the actual UIG fusion level. Preserve its hierarchy
                # gradient instead of attenuating it with downsampled targets.
                union_reliability = 1.0
                intersection_reliability = 1.0

            violation = (
                intersection_reliability
                * F.relu(intersection_probability - seg_probability)
                + union_reliability * F.relu(seg_probability - union_probability)
                + 0.5 * F.relu(intersection_probability - union_probability)
            )
            relevance = torch.maximum(
                torch.maximum(seg_probability, union_probability),
                intersection_probability,
            ).detach()
            normalized_violation = (
                (violation * relevance).sum() / relevance.sum().clamp_min(1e-6)
            )
            hierarchy_loss = hierarchy_loss + float(weight) * normalized_violation

        return self.hierarchy_loss_weight * hierarchy_loss


class nnUNetTrainerSegMambaUIGStableHierarchyCoreExteriorMasked(
    nnUNetTrainerSegMambaUIGStableHierarchy
):
    """Constrain Seg only in target-reliable U/I regions.

    The persistent intersection core may raise Seg confidence, while the
    exterior of the union envelope may suppress false positives. The uncertain
    U-I band is left to the main segmentation loss so thin hemorrhages are not
    removed by an underestimated U branch.
    """

    default_hierarchy_active_scales = 3
    default_hierarchy_lower_weight = 1.0
    default_hierarchy_upper_weight = 1.0
    default_hierarchy_order_weight = 0.25

    def initialize(self):
        self.hierarchy_active_scales = self.default_hierarchy_active_scales
        self.hierarchy_lower_weight = self.default_hierarchy_lower_weight
        self.hierarchy_upper_weight = self.default_hierarchy_upper_weight
        self.hierarchy_order_weight = self.default_hierarchy_order_weight
        super().initialize()
        self.logger.update_config(
            {
                "hierarchy_variant": "core_exterior_masked",
                "hierarchy_active_scales": self.hierarchy_active_scales,
                "hierarchy_scale_weights": [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 0.0],
                "hierarchy_guidance_detached": True,
                "hierarchy_core_mask": "intersection_target_positive",
                "hierarchy_exterior_mask": "union_target_negative_hard_candidates",
                "hierarchy_uncertain_band_weight": 0.0,
                "hierarchy_lower_weight": self.hierarchy_lower_weight,
                "hierarchy_upper_weight": self.hierarchy_upper_weight,
                "hierarchy_order_weight": self.hierarchy_order_weight,
            }
        )
        self.print_to_log_file(
            "Core-Exterior Masked hierarchy: "
            f"weight={self.hierarchy_loss_weight}, active_scales="
            f"{self.hierarchy_active_scales}, weights=4/7,2/7,1/7,0; "
            "detached lower bound in I core, detached upper bound outside U, "
            "no Seg constraint in the uncertain U-I band."
        )

    @staticmethod
    def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (value * weight).sum() / weight.sum().clamp_min(1e-6)

    def _compute_additional_branch_loss(
        self, seg_output, u_output, i_output, u_target=None, i_target=None
    ):
        if u_output is None or i_output is None:
            return nnUNetTrainerSegMambaUIGStable._compute_additional_branch_loss(
                self,
                seg_output,
                u_output,
                i_output,
                u_target=u_target,
                i_target=i_target,
            )
        if u_target is None or i_target is None:
            raise RuntimeError(
                "Core-Exterior Masked hierarchy requires U/I supervision targets."
            )

        seg_outputs = self._as_list(seg_output)
        union_outputs = self._as_list(u_output)
        intersection_outputs = self._as_list(i_output)
        union_targets = self._as_list(u_target)
        intersection_targets = self._as_list(i_target)
        output_counts = {
            len(seg_outputs),
            len(union_outputs),
            len(intersection_outputs),
            len(union_targets),
            len(intersection_targets),
        }
        if len(output_counts) != 1:
            raise RuntimeError(
                "Core-Exterior Masked hierarchy requires matching Seg/U/I outputs "
                "and targets, got "
                f"{len(seg_outputs)}/{len(union_outputs)}/{len(intersection_outputs)}/"
                f"{len(union_targets)}/{len(intersection_targets)}."
            )

        weights = np.zeros(len(seg_outputs), dtype=np.float32)
        active_scales = min(self.hierarchy_active_scales, len(weights))
        weights[:active_scales] = np.asarray(
            [1 / (2 ** index) for index in range(active_scales)], dtype=np.float32
        )
        weights /= weights.sum()

        hierarchy_loss = seg_outputs[0].new_zeros((), dtype=torch.float32)
        for (
            weight,
            seg_logits,
            union_logits,
            intersection_logits,
            union_label,
            intersection_label,
        ) in zip(
            weights,
            seg_outputs,
            union_outputs,
            intersection_outputs,
            union_targets,
            intersection_targets,
        ):
            if weight == 0:
                continue

            seg_probability = self._foreground_probability(seg_logits)
            union_probability = self._foreground_probability(union_logits)
            intersection_probability = self._foreground_probability(intersection_logits)
            union_label = (union_label > 0).float()
            intersection_label = (intersection_label > 0).float()

            union_guidance = union_probability.detach()
            intersection_guidance = intersection_probability.detach()

            core_mask = intersection_label
            lower_violation = F.relu(intersection_guidance - seg_probability)
            lower_loss = self._masked_mean(lower_violation, core_mask)

            # Ignore easy background and normalize only over predicted hard
            # exterior candidates, preventing the large background volume from
            # overwhelming small hemorrhage gradients.
            exterior_mask = 1.0 - union_label
            exterior_relevance = exterior_mask * torch.maximum(
                seg_probability, union_guidance
            ).detach()
            upper_violation = F.relu(seg_probability - union_guidance)
            upper_loss = self._masked_mean(upper_violation, exterior_relevance)

            order_violation = F.relu(
                intersection_probability - union_probability
            )
            order_relevance = torch.maximum(
                intersection_probability, union_probability
            ).detach()
            order_loss = self._masked_mean(order_violation, order_relevance)

            scale_loss = (
                self.hierarchy_lower_weight * lower_loss
                + self.hierarchy_upper_weight * upper_loss
                + self.hierarchy_order_weight * order_loss
            )
            hierarchy_loss = hierarchy_loss + float(weight) * scale_loss

        return self.hierarchy_loss_weight * hierarchy_loss


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
