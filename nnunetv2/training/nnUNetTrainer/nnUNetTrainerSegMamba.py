# -*- coding: utf-8 -*-
import shutil
import numpy as np
import torch
import torch.distributed as dist
from batchgenerators.utilities.file_and_folder_operations import join
from tqdm import tqdm
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerSegMamba(nnUNetTrainer):
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

    def initialize(self):
        ### 🚀 核心参数自定义配置区（可在此自由修改） 🚀 ###
        # 1. 目标学习率 (nnU-Net 默认是 0.01)
        self.initial_lr = 3e-3  
        # 2. 训练总轮数
        self.num_epochs = 500  
        # 3. 梯度裁剪最大范数 (设为 <= 0 则关闭)
        self.custom_max_grad_norm = 12.0  
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
                "batch_size": self.configuration_manager.batch_size
            }
            self.logger.update_config({"hparas": logger_config_hparas})
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
            loss = DC_and_BCE_loss({}, {"batch_dice": self.configuration_manager.batch_dice, "do_bg": True, "smooth": 1e-5, "ddp": self.is_ddp}, use_ignore_label=self.label_manager.ignore_label is not None, dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({"batch_dice": self.configuration_manager.batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp}, {}, weight_ce=1, weight_dice=1, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp: weights[-1] = 1e-6
            else: weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

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
        if isinstance(output, (list, tuple)): output = [torch.clamp(o, min=-45.0, max=45.0) for o in output]
        else: output = torch.clamp(output, min=-45.0, max=45.0)

        l = self.loss(output, target)
        if torch.isnan(l) or torch.isinf(l): return {'loss': 0.0, 'train_dice': 0.0}
        l.backward()
        
        with torch.no_grad():
            main_output = output[0] if isinstance(output, (list, tuple)) else output
            main_target = target[0] if isinstance(target, (list, tuple)) else target
            num_classes = main_output.shape[1]
            output_seg = main_output.argmax(1)
            output_onehot = torch.nn.functional.one_hot(output_seg, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            target_onehot = torch.nn.functional.one_hot(main_target.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            intersect = torch.sum(output_onehot[:, 1:] * target_onehot[:, 1:], dim=(2, 3, 4))
            denom = torch.sum(output_onehot[:, 1:] + target_onehot[:, 1:], dim=(2, 3, 4))
            step_dice = (2. * intersect / (denom + 1e-5)).mean().cpu().item()
            if np.isnan(step_dice): step_dice = 0.0

        if self.custom_max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.custom_max_grad_norm)
        self.optimizer.step()
        return {'loss': l.detach().cpu().item(), 'train_dice': step_dice}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in batch['target']] if isinstance(batch['target'], (list, tuple)) else batch['target'].to(self.device, non_blocking=True)

        output = self.network(data)
        del data
        if self.enable_deep_supervision and isinstance(output, (list, tuple)):
            l = self.loss(output, target)
            output = output[0]
            target = target[0]
        else:
            base_loss = self.loss.loss if hasattr(self.loss, "loss") else self.loss
            l = base_loss(output, target[0] if isinstance(target, (list, tuple)) else target)
            if isinstance(target, (list, tuple)): target = target[0]

        num_classes = output.shape[1]
        output_seg = output.argmax(1)
        output_onehot = torch.nn.functional.one_hot(output_seg, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        target_onehot = torch.nn.functional.one_hot(target.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        
        tp_hard = torch.sum(output_onehot * target_onehot, dim=(2, 3, 4)).detach().cpu().numpy()
        fp_hard = torch.sum(output_onehot * (1. - target_onehot), dim=(2, 3, 4)).detach().cpu().numpy()
        fn_hard = torch.sum((1. - output_onehot) * target_onehot, dim=(2, 3, 4)).detach().cpu().numpy()
        return {'loss': l.detach().cpu().item(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

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
            except Exception as e:
                self.print_to_log_file(f"⚠️ 指标实时汇总器遇到错误: {str(e)}", also_print_to_console=True)

    def on_train_end(self):
        super().on_train_end()
        is_main_process = (not self.is_ddp) or (dist.get_rank() == 0)
        if not is_main_process:
            return
        if self.best_val_checkpoint_file is None:
            return
        try:
            shutil.copyfile(self.best_val_checkpoint_file, join(self.output_folder, "checkpoint_final.pth"))
        except FileNotFoundError:
            self.print_to_log_file(
                "⚠️ 未找到 checkpoint_best_val.pth，保留原始 checkpoint_final.pth。",
                also_print_to_console=True,
            )
        else:
            self.print_to_log_file(
                f"🏁 已将验证集最优权重同步为最终 checkpoint: {self.best_val_checkpoint_file} -> checkpoint_final.pth",
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
