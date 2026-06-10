# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerSegMamba(nnUNetTrainer):
    """
    自定义 SegMamba Trainer：
    - 支持 deep_supervision=True|False
    - 训练 & 验证循环显示 tqdm
    - DDP 多卡模式下仅 Rank 0 显示
    """

    def initialize(self):
        # 预先设置是否开启 deep supervision
        # 从 plans 的 arch_kwargs 中读取
        self.enable_deep_supervision = self.configuration_manager.network_arch_init_kwargs.get(
            "deep_supervision", False
        )
        super().initialize()

    def _do_i_compile(self):
        # SegMamba 网络可能不适合 torch.compile
        return False

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        在 DDP 或 Compiler 包装模型里面设置 deep_supervision
        """
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        # 先试 decoder 再 network 自身
        if hasattr(mod, "decoder"):
            mod.decoder.deep_supervision = enabled
        if hasattr(mod, "deep_supervision"):
            mod.deep_supervision = enabled

    def _build_loss(self):
        """
        构造 Loss，可根据 deep_supervision 自动包装
        """
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        # 如果开启 deep supervision，则包装 multi-output loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def run_training(self):
        """
        重写训练循环，支持 tqdm
        并兼容 DDP rank 0 显示
        """
        self.on_train_start()

        # 我们只在主进程显示 tqdm 以避免多卡刷屏
        is_main_process = (not self.is_ddp) or (dist.get_rank() == 0)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            train_outputs = []

            # 训练 tqdm
            if is_main_process:
                pbar = tqdm(range(self.num_iterations_per_epoch),
                            desc=f"Train Epoch {epoch}",
                            dynamic_ncols=True)
            else:
                pbar = range(self.num_iterations_per_epoch)

            for _ in pbar:
                batch = next(self.dataloader_train)
                out = self.train_step(batch)
                train_outputs.append(out)

                if is_main_process and isinstance(out, dict) and "loss" in out:
                    pbar.set_postfix({
                        "loss": f"{out['loss']:.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })

            self.on_train_epoch_end(train_outputs)

            # 验证 tqdm
            val_outputs = []

            if is_main_process:
                pbar_val = tqdm(range(self.num_val_iterations_per_epoch),
                                desc=f"Val Epoch {epoch}",
                                dynamic_ncols=True)
            else:
                pbar_val = range(self.num_val_iterations_per_epoch)

            for _ in pbar_val:
                batch = next(self.dataloader_val)
                out_v = self.validation_step(batch)
                val_outputs.append(out_v)

            self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()

        self.on_train_end()