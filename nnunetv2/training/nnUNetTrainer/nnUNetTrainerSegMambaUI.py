# -*- coding: utf-8 -*-
import math
import shutil
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join
from tqdm import tqdm
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs


class CosineAnnealingWarmRestartsWithMultipliers:
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        first_cycle_steps: int = 50,
        cycle_mult: int = 2,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.min_lr = min_lr
        self.ctr = 0
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        t_cur, t_i = self._get_cycle_position(current_step)
        cosine = (1 + math.cos(math.pi * t_cur / t_i)) / 2
        base_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine
        self._last_lr = []
        for param_group in self.optimizer.param_groups:
            multiplier = param_group.get("lr_multiplier", 1.0)
            group_lr = base_lr * multiplier
            param_group["lr"] = group_lr
            self._last_lr.append(param_group["lr"])

    def get_last_lr(self):
        return self._last_lr

    def _get_cycle_position(self, current_step: int):
        current_step = max(0, int(current_step))
        if self.cycle_mult == 1:
            return current_step % self.first_cycle_steps, self.first_cycle_steps

        cycle = 0
        cycle_steps = self.first_cycle_steps
        remaining = current_step
        while remaining >= cycle_steps:
            remaining -= cycle_steps
            cycle += 1
            cycle_steps *= self.cycle_mult
        return remaining, cycle_steps


class nnUNetTrainerSegMambaUI(nnUNetTrainer):
    """
    针对 SegMambaV2 深度定制的单多卡接力版 Trainer（进度精细化看板完结版）：
    1. 彻底关闭混合精度(AMP)，全流程纯 FP32 规避 Mamba SSM 架构的 NaN 溢出。
    2. 允许在中途断点续训(--c)时，让修改的学习率和总轮数立刻生效。
    3. 集成 12.0 模长梯度裁剪数值安全防线。
    4. 彻底对齐官方 get_value 接口，完爆 MetaLogger 属性缺失报错。
    5. 【全新修改】动态拼接进度条描述符，让 Train/Val 的 tqdm 进度条同时高亮显示“当前 Epoch / 总 Epoch”。
    6. 默认不再冻结前 15 轮骨干，避免限制 SegMamba 的早期收敛和最终泛化上限。
    7. 仅保留 4 个尺度进行多尺度 Loss 评估（彻底剔除 dec0 深监督信号）。
    8. 每个 Epoch 结束时，另起一行，以精美树状格式统一汇总当前 Epoch 的两端完整数据（Train/Val Loss & Dice）及历史最佳纪录。
    """

    default_union_loss_weight = 0.3
    default_intersection_loss_weight = 0.3
    default_auxiliary_lr_multiplier = 3.0
    dice_loss_class = MemoryEfficientSoftDiceLoss

    def initialize(self):
        ### 🚀 核心参数自定义配置区（可在此自由修改） 🚀 ###
        # 1. 目标学习率 (nnU-Net 默认是 0.01)
        self.initial_lr = 3e-3  
        # 2. 训练总轮数
        self.num_epochs = 500  
        # 3. 梯度裁剪最大范数 (设为 <= 0 则关闭)
        self.custom_max_grad_norm = 12.0  
        # 4. U/I 辅助分支 loss 权重
        self.union_loss_weight = self.default_union_loss_weight
        self.intersection_loss_weight = self.default_intersection_loss_weight
        # 5. U/I 分支没有直接预训练，使用更高学习率加速适配
        self.auxiliary_lr_multiplier = self.default_auxiliary_lr_multiplier
        # 6. CosineAnnealingWarmRestarts 参数
        self.lr_restart_t0 = 50
        self.lr_restart_t_mult = 2
        self.lr_min = 1e-6
        ################################################

        self.best_epoch = 0
        self.best_val_dice = 0.0
        self.epoch_train_dice_list = []
        self.pretrain_flag = False
        self.freeze_backbone_epochs = 0
        self.best_val_checkpoint_file = None

        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)

            self.network = self.build_network_architecture(
                self.plans_manager,
                self.configuration_manager,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)

            self.loss = self._build_loss()
            self.ui_loss = self._build_binary_aux_loss()
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True

            logger_config_hparas = {
                "initial_lr": self.initial_lr, "weight_decay": self.weight_decay,
                "oversample_foreground_percent": self.oversample_foreground_percent,
                "probabilistic_oversampling": self.probabilistic_oversampling,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs, "enable_deep_supervision": self.enable_deep_supervision,
                "batch_size": self.configuration_manager.batch_size,
                "union_loss_weight": self.union_loss_weight,
                "intersection_loss_weight": self.intersection_loss_weight,
                "auxiliary_lr_multiplier": self.auxiliary_lr_multiplier,
                "lr_scheduler": "CosineAnnealingWarmRestarts",
                "lr_restart_t0": self.lr_restart_t0,
                "lr_restart_t_mult": self.lr_restart_t_mult,
                "lr_min": self.lr_min,
            }
            self.logger.update_config({"hparas": logger_config_hparas})
            self._ensure_aux_logger_keys()
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized.")

        self.grad_scaler = None

        pretrained_path = self.configuration_manager.network_arch_init_kwargs.get('pretrained_path', None)
        self.pretrain_flag = pretrained_path
        if pretrained_path is None:
            self.print_to_log_file("pretrained_path is none, not loading any pretrained weights. If you want to load pretrained weights, please set the 'pretrained_path' in the network_arch_init_kwargs of the configuration_manager.")
        else:
            self.print_to_log_file(f"pretrained_path is {pretrained_path}, loading pretrained weights from this path.")
            self.pretrain_flag = True
        self.best_val_checkpoint_file = join(self.output_folder, "checkpoint_best_val.pth")
        if self.current_epoch == 0 and pretrained_path is not None:
            mod = self.network
            if isinstance(mod, DDP):
                mod = mod.module
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            if hasattr(mod, "load_from"):
                mod.load_from(pretrained_path)

    def _get_deep_supervision_scales(self):
        return [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]]

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({}, {"batch_dice": self.configuration_manager.batch_dice, "do_bg": True, "smooth": 1e-5, "ddp": self.is_ddp}, use_ignore_label=self.label_manager.ignore_label is not None, dice_class=self.dice_loss_class)
        else:
            loss = DC_and_CE_loss({"batch_dice": self.configuration_manager.batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp}, {}, weight_ce=1, weight_dice=1, ignore_label=self.label_manager.ignore_label, dice_class=self.dice_loss_class)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp: weights[-1] = 1e-6
            else: weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def configure_optimizers(self):
        aux_prefixes = ("u_decoder", "i_decoder", "u_out", "i_out")
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

    def _build_binary_aux_loss(self):
        loss = DC_and_CE_loss(
            {"batch_dice": self.configuration_manager.batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=None,
            dice_class=self.dice_loss_class,
        )
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp:
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _ensure_aux_logger_keys(self):
        keys = (
            "train_losses_seg", "train_losses_u", "train_losses_i",
            "val_losses_seg", "val_losses_u", "val_losses_i",
            "train_dice_seg", "train_dice_u", "train_dice_i",
            "val_dice_seg", "val_dice_u", "val_dice_i",
        )
        local_logging = self.logger.local_logger.my_fantastic_logging
        for key in keys:
            local_logging.setdefault(key, [])

    @staticmethod
    def _as_list(x):
        return list(x) if isinstance(x, (list, tuple)) else [x]

    @staticmethod
    def _first_output(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    @staticmethod
    def _clamp_branch(x):
        if isinstance(x, (list, tuple)):
            return [torch.clamp(i, min=-45.0, max=45.0) for i in x]
        return torch.clamp(x, min=-45.0, max=45.0)

    @staticmethod
    def _split_network_output(output):
        if isinstance(output, dict):
            return output.get("seg"), output.get("u"), output.get("i")
        return output, None, None

    def _compute_additional_branch_loss(
        self, seg_output, u_output, i_output, u_target=None, i_target=None
    ):
        reference = self._first_output(seg_output)
        return reference.new_zeros(())

    def _make_union_intersection_targets(self, target):
        target_list = self._as_list(target)
        base_target = target_list[0]
        foreground = (base_target > 0).float()
        if foreground.ndim != 5:
            raise RuntimeError(f"U/I supervision expects Bx1xDxHxW targets, got {tuple(foreground.shape)}")

        padded = F.pad(foreground, (0, 0, 0, 0, 1, 1), mode="replicate")
        union_full = F.max_pool3d(padded, kernel_size=(3, 1, 1), stride=1)
        intersection_full = -F.max_pool3d(-padded, kernel_size=(3, 1, 1), stride=1)

        union_targets = []
        intersection_targets = []
        for t in target_list:
            shape = t.shape[2:]
            if tuple(shape) == tuple(union_full.shape[2:]):
                union_targets.append(union_full.long())
                intersection_targets.append(intersection_full.long())
            else:
                union_targets.append(F.interpolate(union_full, size=shape, mode="nearest").long())
                intersection_targets.append(F.interpolate(intersection_full, size=shape, mode="nearest").long())
        if isinstance(target, (list, tuple)):
            return union_targets, intersection_targets
        return union_targets[0], intersection_targets[0]

    @staticmethod
    def _mean_dice_from_logits(logits, target, include_background=False):
        logits = logits.detach()
        target = target.detach().long()
        num_classes = logits.shape[1]
        pred = logits.argmax(1)
        target = target.squeeze(1)
        pred_onehot = torch.nn.functional.one_hot(pred, num_classes=num_classes).movedim(-1, 1).float()
        target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1).float()
        start_idx = 0 if include_background else 1
        intersect = torch.sum(pred_onehot[:, start_idx:] * target_onehot[:, start_idx:], dim=tuple(range(2, pred_onehot.ndim)))
        denom = torch.sum(pred_onehot[:, start_idx:] + target_onehot[:, start_idx:], dim=tuple(range(2, pred_onehot.ndim)))
        dice = (2.0 * intersect / (denom + 1e-5)).mean()
        return 0.0 if torch.isnan(dice) else dice.cpu().item()

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network
        if isinstance(mod, DDP): mod = mod.module
        if isinstance(mod, OptimizedModule): mod = mod._orig_mod
        if hasattr(mod, "deep_supervision"): mod.deep_supervision = enabled

    def train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        data = batch['data'].to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in batch['target']] if isinstance(batch['target'], (list, tuple)) else batch['target'].to(self.device, non_blocking=True)

        output = self.network(data)
        del data
        seg_output, u_output, i_output = self._split_network_output(output)
        seg_output = self._clamp_branch(seg_output)
        u_output = self._clamp_branch(u_output) if u_output is not None else None
        i_output = self._clamp_branch(i_output) if i_output is not None else None

        u_target, i_target = self._make_union_intersection_targets(target)
        loss_seg = self.loss(seg_output, target)
        loss_u = self.ui_loss(u_output, u_target) if u_output is not None else loss_seg.new_tensor(0.0)
        loss_i = self.ui_loss(i_output, i_target) if i_output is not None else loss_seg.new_tensor(0.0)
        loss_additional = self._compute_additional_branch_loss(
            seg_output, u_output, i_output, u_target=u_target, i_target=i_target
        )
        l = (
            loss_seg
            + self.union_loss_weight * loss_u
            + self.intersection_loss_weight * loss_i
            + loss_additional
        )
        if torch.isnan(l) or torch.isinf(l):
            return {
                'loss': 0.0, 'loss_seg': 0.0, 'loss_u': 0.0, 'loss_i': 0.0, 'loss_additional': 0.0,
                'train_dice': 0.0, 'train_dice_seg': 0.0, 'train_dice_u': 0.0, 'train_dice_i': 0.0,
            }
        l.backward()
        
        with torch.no_grad():
            main_output = self._first_output(seg_output)
            main_target = target[0] if isinstance(target, (list, tuple)) else target
            step_dice = self._mean_dice_from_logits(main_output, main_target)
            u_dice = self._mean_dice_from_logits(self._first_output(u_output), self._first_output(u_target)) if u_output is not None else 0.0
            i_dice = self._mean_dice_from_logits(self._first_output(i_output), self._first_output(i_target)) if i_output is not None else 0.0

        if self.custom_max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.custom_max_grad_norm)
        self.optimizer.step()
        return {
            'loss': l.detach().cpu().item(),
            'loss_seg': loss_seg.detach().cpu().item(),
            'loss_u': loss_u.detach().cpu().item(),
            'loss_i': loss_i.detach().cpu().item(),
            'loss_additional': loss_additional.detach().cpu().item(),
            'train_dice': step_dice,
            'train_dice_seg': step_dice,
            'train_dice_u': u_dice,
            'train_dice_i': i_dice,
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in batch['target']] if isinstance(batch['target'], (list, tuple)) else batch['target'].to(self.device, non_blocking=True)

        output = self.network(data)
        del data
        seg_output, u_output, i_output = self._split_network_output(output)
        u_target, i_target = self._make_union_intersection_targets(target)
        loss_additional = self._compute_additional_branch_loss(
            seg_output, u_output, i_output, u_target=u_target, i_target=i_target
        )

        if self.enable_deep_supervision and isinstance(seg_output, (list, tuple)):
            loss_seg = self.loss(seg_output, target)
            loss_u = self.ui_loss(u_output, u_target) if u_output is not None else loss_seg.new_tensor(0.0)
            loss_i = self.ui_loss(i_output, i_target) if i_output is not None else loss_seg.new_tensor(0.0)
            l = (
                loss_seg
                + self.union_loss_weight * loss_u
                + self.intersection_loss_weight * loss_i
                + loss_additional
            )
            output = seg_output[0]
            target_main = target[0]
            u_output_main = u_output[0] if u_output is not None else None
            i_output_main = i_output[0] if i_output is not None else None
            u_target_main = u_target[0]
            i_target_main = i_target[0]
        else:
            base_loss = self.loss.loss if hasattr(self.loss, "loss") else self.loss
            base_ui_loss = self.ui_loss.loss if hasattr(self.ui_loss, "loss") else self.ui_loss
            target_main = target[0] if isinstance(target, (list, tuple)) else target
            u_target_main = u_target[0] if isinstance(u_target, (list, tuple)) else u_target
            i_target_main = i_target[0] if isinstance(i_target, (list, tuple)) else i_target
            loss_seg = base_loss(seg_output, target_main)
            loss_u = base_ui_loss(u_output, u_target_main) if u_output is not None else loss_seg.new_tensor(0.0)
            loss_i = base_ui_loss(i_output, i_target_main) if i_output is not None else loss_seg.new_tensor(0.0)
            l = (
                loss_seg
                + self.union_loss_weight * loss_u
                + self.intersection_loss_weight * loss_i
                + loss_additional
            )
            output = seg_output
            u_output_main = u_output
            i_output_main = i_output

        num_classes = output.shape[1]
        output_seg = output.argmax(1)
        output_onehot = torch.nn.functional.one_hot(output_seg, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        target_onehot = torch.nn.functional.one_hot(target_main.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        
        tp_hard = torch.sum(output_onehot * target_onehot, dim=(0, 2, 3, 4)).detach().cpu().numpy()
        fp_hard = torch.sum(output_onehot * (1. - target_onehot), dim=(0, 2, 3, 4)).detach().cpu().numpy()
        fn_hard = torch.sum((1. - output_onehot) * target_onehot, dim=(0, 2, 3, 4)).detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        seg_dice = self._mean_dice_from_logits(output, target_main)
        u_dice = self._mean_dice_from_logits(u_output_main, u_target_main) if u_output_main is not None else 0.0
        i_dice = self._mean_dice_from_logits(i_output_main, i_target_main) if i_output_main is not None else 0.0
        return {
            'loss': l.detach().cpu().item(),
            'loss_seg': loss_seg.detach().cpu().item(),
            'loss_u': loss_u.detach().cpu().item(),
            'loss_i': loss_i.detach().cpu().item(),
            'loss_additional': loss_additional.detach().cpu().item(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'val_dice_seg': seg_dice,
            'val_dice_u': u_dice,
            'val_dice_i': i_dice,
        }

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self._ensure_aux_logger_keys()
        outputs = collate_outputs(train_outputs)
        for output_key, log_key in (
            ("loss_seg", "train_losses_seg"),
            ("loss_u", "train_losses_u"),
            ("loss_i", "train_losses_i"),
            ("train_dice_seg", "train_dice_seg"),
            ("train_dice_u", "train_dice_u"),
            ("train_dice_i", "train_dice_i"),
        ):
            if output_key in outputs:
                self.logger.log(log_key, float(np.mean(outputs[output_key])), self.current_epoch)

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        self._ensure_aux_logger_keys()
        outputs = collate_outputs(val_outputs)
        for output_key, log_key in (
            ("loss_seg", "val_losses_seg"),
            ("loss_u", "val_losses_u"),
            ("loss_i", "val_losses_i"),
            ("val_dice_seg", "val_dice_seg"),
            ("val_dice_u", "val_dice_u"),
            ("val_dice_i", "val_dice_i"),
        ):
            if output_key in outputs:
                self.logger.log(log_key, float(np.mean(outputs[output_key])), self.current_epoch)

    def run_training(self):
        import warnings
        warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`", category=UserWarning)
        self.on_train_start()

        if self.freeze_backbone_epochs > 0 and self.current_epoch < self.freeze_backbone_epochs:
            self.print_to_log_file(
                f"🔒 [骨干冻结] 当前属于前 {self.freeze_backbone_epochs} 轮冻结阶段，锁定 SegMamba 骨干权重...",
                also_print_to_console=True,
            )
            mod = self.network.module if self.is_ddp else self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            for name, param in mod.named_parameters():
                if "vit" in name or "decoder" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            self.print_to_log_file("🔓 [全网训练] 当前不启用骨干冻结，直接训练全网参数...", also_print_to_console=True)
            for param in self.network.parameters():
                param.requires_grad = True

        is_main_process = (not self.is_ddp) or (dist.get_rank() == 0)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()

            if self.freeze_backbone_epochs > 0 and epoch == self.freeze_backbone_epochs:
                self.print_to_log_file(
                    f"🔓 [骨干解冻] 冻结阶段结束，全面解冻全网所有参数...",
                    also_print_to_console=True,
                )
                for param in self.network.parameters():
                    param.requires_grad = True

            train_outputs = []
            self.epoch_train_dice_list.clear()
            
            if is_main_process:
                # 🌟【核心改动点】将原本的 "Train Epoch 0" 改为具有极强进度感的 "Train Epoch 0/250" 看板形式
                with tqdm(desc=f" 🚀 Train Epoch {epoch}", total=self.num_iterations_per_epoch, ncols=110) as pbar:
                    for _ in range(self.num_iterations_per_epoch):
                        res = self.train_step(next(self.dataloader_train))
                        train_outputs.append(res)
                        self.epoch_train_dice_list.append(res['train_dice'])
                        pbar.set_postfix(loss=f"{res['loss']:.4f}", tr_dice=f"{res['train_dice']:.4f}")
                        pbar.update(1)
            else:
                for _ in range(self.num_iterations_per_epoch):
                    res = self.train_step(next(self.dataloader_train))
                    train_outputs.append(res)
                    self.epoch_train_dice_list.append(res['train_dice'])

            self.on_train_epoch_end(train_outputs)
            
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                if is_main_process:
                    # 🌟【核心改动点】将原本的 "Val Epoch 0" 改为 "Val Epoch 0/250" 对齐看板形式
                    with tqdm(desc=f" 👁️ Val Epoch {epoch}", total=self.num_val_iterations_per_epoch, ncols=110) as pbar:
                        for _ in range(self.num_val_iterations_per_epoch):
                            val_outputs.append(self.validation_step(next(self.dataloader_val)))
                            pbar.update(1)
                else:
                    for _ in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            
            self.on_epoch_end()
        self.on_train_end()

    def _plot_aux_progress(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            self.print_to_log_file(f"Could not import matplotlib for aux progress plot: {e}")
            return

        logs = self.logger.local_logger.my_fantastic_logging
        groups = [
            (
                "Loss",
                (
                    "train_losses_seg", "val_losses_seg",
                    "train_losses_u", "val_losses_u",
                    "train_losses_i", "val_losses_i",
                    "train_losses_hierarchy", "val_losses_hierarchy",
                ),
            ),
            ("Dice", ("train_dice_seg", "val_dice_seg", "train_dice_u", "val_dice_u", "train_dice_i", "val_dice_i")),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=120)
        for ax, (title, keys) in zip(axes, groups):
            for key in keys:
                values = logs.get(key, [])
                if len(values) > 0:
                    ax.plot(values, label=key)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "progress_seg_u_i.png"))
        plt.close(fig)

    def on_epoch_end(self):
        epoch_idx = self.current_epoch
        super().on_epoch_end()
        is_main_process = (not self.is_ddp) or (dist.get_rank() == 0)
        if is_main_process:
            try:
                train_losses = self.logger.get_value('train_losses', step=-1)
                val_losses = self.logger.get_value('val_losses', step=-1)
                current_val_dice = self.logger.get_value('mean_fg_dice', step=-1)
                mean_train_dice = np.mean(self.epoch_train_dice_list) if len(self.epoch_train_dice_list) > 0 else 0.0
                if current_val_dice >= self.best_val_dice:
                    self.best_val_dice = current_val_dice
                    self.best_epoch = epoch_idx
                    if self.best_val_checkpoint_file is not None:
                        self.save_checkpoint(self.best_val_checkpoint_file)

                self.print_to_log_file("\n" + "="*85, also_print_to_console=True)
                self.print_to_log_file(
                    f"📊 [Epoch {epoch_idx}/{self.num_epochs} 训练/验证全套指标独立总结]:\n"
                    f"   ├─ 🚀 训练集  (Train)  ──> Loss: {train_losses:.4f}  |  Mean Dice: {mean_train_dice:.4f}\n"
                    f"   ├─ 👁️ 验证集  (Val)   ──> Loss: {val_losses:.4f}  |  Mean Dice: {current_val_dice:.4f}\n"
                    f"   └─ 🏆 历史最佳 (Best)   ──> Best Val Dice: {self.best_val_dice:.4f} (于第 {self.best_epoch} 个 Epoch 获得)",
                    also_print_to_console=True
                )
                self.print_to_log_file("="*85 + "\n", also_print_to_console=True)
                self._plot_aux_progress()
            except Exception as e:
                self.print_to_log_file(f"⚠️ 指标实时汇总器遇到错误: {str(e)}", also_print_to_console=True)

    def on_train_end(self):
        super().on_train_end()
        self._synchronize_best_checkpoint_for_final_validation()

    def _synchronize_best_checkpoint_for_final_validation(self):
        """Make final validation and later inference use the same best weights."""
        is_main_process = (not self.is_ddp) or (dist.get_rank() == 0)
        if self.best_val_checkpoint_file is None:
            return

        copied_best = False
        if is_main_process:
            try:
                shutil.copyfile(
                    self.best_val_checkpoint_file,
                    join(self.output_folder, "checkpoint_final.pth"),
                )
            except FileNotFoundError:
                self.print_to_log_file(
                    "⚠️ 未找到 checkpoint_best_val.pth，保留原始 checkpoint_final.pth。",
                    also_print_to_console=True,
                )
            else:
                copied_best = True

        if self.is_ddp:
            copied_best_tensor = torch.tensor(
                [int(copied_best)], dtype=torch.uint8, device=self.device
            )
            dist.broadcast(copied_best_tensor, src=0)
            copied_best = bool(copied_best_tensor.item())

        if copied_best:
            final_checkpoint = join(self.output_folder, "checkpoint_final.pth")
            self.load_checkpoint(final_checkpoint)
            if is_main_process:
                self.print_to_log_file(
                    "🏁 已将验证集最优权重同步为最终 checkpoint 并重新加载到所有 "
                    f"DDP rank: {self.best_val_checkpoint_file} -> {final_checkpoint}",
                    also_print_to_console=True,
                )

    def mfa_checkpoint_save_loads(self, lr, num_epochs):
        self.initial_lr = lr
        self.num_epochs = num_epochs
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups: param_group['initial_lr'] = lr

    def load_checkpoint(self, checkpoint_file: str):
        super().load_checkpoint(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
        if 'best_epoch' in checkpoint: self.best_epoch = checkpoint['best_epoch']
        if 'best_val_dice' in checkpoint: self.best_val_dice = checkpoint['best_val_dice']

    def save_checkpoint(self, filename: str):
        super().save_checkpoint(filename)
        is_main_process = (not self.is_ddp) or (dist.get_rank() == 0)
        if is_main_process and filename.endswith('.pth'):
            checkpoint = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
            checkpoint['best_epoch'] = self.best_epoch
            checkpoint['best_val_dice'] = self.best_val_dice
            torch.save(checkpoint, filename)
