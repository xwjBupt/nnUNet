import math

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegMambaUI import nnUNetTrainerSegMambaUI


class nnUNetTrainerSegMambaUIAnisoPhysicalSDM(nnUNetTrainerSegMambaUI):
    """Train anisotropic SegMamba UI with a physical core/envelope target."""

    default_union_loss_weight = 0.2
    default_intersection_loss_weight = 0.1
    default_auxiliary_lr_multiplier = 1.0
    physical_band_radius_mm = 2.0

    def initialize(self):
        self.physical_band_radius_mm = float(
            self.configuration_manager.configuration.get(
                "physical_band_radius_mm", self.physical_band_radius_mm
            )
        )
        self._physical_kernel_cache = {}
        super().initialize()
        spacing = tuple(float(v) for v in self.configuration_manager.spacing)
        kernel_shape = self._physical_kernel_shape(spacing)
        self.logger.update_config(
            {
                "physical_band_radius_mm": self.physical_band_radius_mm,
                "physical_band_spacing_mm": spacing,
                "physical_band_kernel_shape": kernel_shape,
            }
        )
        self.print_to_log_file(
            "Physical U/I supervision: "
            f"radius={self.physical_band_radius_mm:.2f} mm, spacing={spacing}, "
            f"kernel_shape={kernel_shape}."
        )

    def _get_deep_supervision_scales(self):
        strides = np.asarray(
            self.configuration_manager.network_arch_init_kwargs["strides"], dtype=np.float64
        )
        if strides.shape != (4, 3):
            raise RuntimeError(f"Expected four 3D SegMamba strides, got {strides.tolist()}.")
        cumulative = np.cumprod(strides, axis=0)
        return [[1.0, 1.0, 1.0], *(1.0 / cumulative[:3]).tolist()]

    def _physical_kernel_shape(self, spacing):
        radii = [int(math.floor(self.physical_band_radius_mm / value)) for value in spacing]
        return tuple(2 * radius + 1 for radius in radii)

    def _physical_kernel(self, spacing, device, dtype):
        key = (tuple(spacing), str(device), dtype)
        cached = self._physical_kernel_cache.get(key)
        if cached is not None:
            return cached

        voxel_radii = [int(math.floor(self.physical_band_radius_mm / value)) for value in spacing]
        axes = [
            torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) * float(axis_spacing)
            for radius, axis_spacing in zip(voxel_radii, spacing)
        ]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        kernel = (
            zz.square() + yy.square() + xx.square()
            <= self.physical_band_radius_mm ** 2 + 1e-6
        ).to(dtype=dtype)
        kernel = kernel[None, None]
        self._physical_kernel_cache[key] = kernel
        return kernel

    def _make_union_intersection_targets(self, target):
        target_list = self._as_list(target)
        base_target = target_list[0]
        foreground = (base_target > 0).float()
        if foreground.ndim != 5:
            raise RuntimeError(
                f"Physical U/I supervision expects Bx1xDxHxW targets, got {tuple(foreground.shape)}"
            )

        spacing = tuple(float(v) for v in self.configuration_manager.spacing)
        kernel = self._physical_kernel(spacing, foreground.device, foreground.dtype)
        radius_d, radius_h, radius_w = (int((size - 1) // 2) for size in kernel.shape[2:])
        padded = F.pad(
            foreground,
            (radius_w, radius_w, radius_h, radius_h, radius_d, radius_d),
            mode="replicate",
        )
        neighborhood_count = F.conv3d(padded, kernel)
        union_full = (neighborhood_count > 0).float()
        intersection_full = (neighborhood_count >= kernel.sum() - 0.5).float()

        union_targets = []
        intersection_targets = []
        for deep_supervision_target in target_list:
            shape = deep_supervision_target.shape[2:]
            if tuple(shape) == tuple(union_full.shape[2:]):
                union_targets.append(union_full.long())
                intersection_targets.append(intersection_full.long())
            else:
                union_targets.append(F.interpolate(union_full, size=shape, mode="nearest").long())
                intersection_targets.append(
                    F.interpolate(intersection_full, size=shape, mode="nearest").long()
                )

        if isinstance(target, (list, tuple)):
            return union_targets, intersection_targets
        return union_targets[0], intersection_targets[0]
