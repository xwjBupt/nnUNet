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
    针对 SegMambaV2 深度定制的完整版 Trainer：
    1. 彻底关闭混合精度(AMP)，前向与反向完全运行在纯 FP32 下，根治 Mamba SSM 架构的 NaN 溢出问题。
    2. 攻克 PyTorch 状态恢复限制：允许在中途断点续训(--c)时，强行让代码里修改的学习率和总轮数立刻生效。
    3. 集成梯度裁剪(Gradient Clipping)作为核心数值安全防线。
    4. 完美重写训练与验证循环，支持单卡/多卡(DDP Rank 0) tqdm 进度条，且不破坏 nnU-Net 原生的日志与画图机制。
    """

    def initialize(self):
        ### 🚀 核心参数自定义配置区（可在此自由修改） 🚀 ###
        # 1. 目标学习率 (nnU-Net 默认是 0.01)
        # 提示：鉴于 SegMamba 在 216 轮遇到过震荡，若使用纯 FP32 恢复，建议设为 1e-3 (0.001) 或 5e-4 观察
        self.custom_initial_lr = 1e-3  
        
        # 2. 权重衰减系数 (nnU-Net 默认是 3e-5)
        self.custom_weight_decay = 3e-5  
        
        # 3. 期望的最终总训练轮数 (nnU-Net 默认是 1000)
        # 即使断点恢复，训练达到这个设定的轮数时系统就会安全结束并保存 final 权重
        self.custom_num_epochs = 500  
        
        # 4. 每轮训练迭代次数 (nnU-Net 默认是 250)
        self.custom_num_iterations_per_epoch = 250  
        
        # 5. 梯度裁剪最大模长 (设为 0 则关闭裁剪，推荐 12.0)
        self.custom_max_grad_norm = 12.0
        #####################################################

        # 从 plans 的 arch_kwargs 中读取是否开启 deep supervision
        self.enable_deep_supervision = self.configuration_manager.network_arch_init_kwargs.get(
            "deep_supervision", False
        )
        
        # 调用基类 initialize 完成网络架构搭建与数据流加载
        super().initialize()
        
        # 【关键改动】覆盖基类，强制关闭原生的自动混合精度 AMP 缩放器，开启纯 FP32 模式
        self.grad_scaler = None
        
        # 将自定义的训练轮数和每轮迭代步数应用到系统变量中
        self.num_epochs = self.custom_num_epochs
        self.num_iterations_per_epoch = self.custom_num_iterations_per_epoch

    def configure_optimizers(self):
        """
        重写优化器和学习率调度器配置，将自定义的参数传入
        默认仍采用 nnU-Net 标志性的 SGD + PolyLR 衰减策略
        """
        optimizer = torch.optim.SGD(
            self.network.parameters(), 
            lr=self.custom_initial_lr,             # 应用自定义初始学习率
            momentum=0.99, 
            weight_decay=self.custom_weight_decay, # 应用自定义权重衰减
            nesterov=True
        )
        
        # 采用 nnU-Net 原生的 Poly 衰减学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: (1 - epoch / self.num_epochs) ** 0.9
        )
        
        return optimizer, lr_scheduler

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """
        重写基类的载入检查点函数：
        在历史的 optimizer_state 被 load 进来之后，利用我们自定义的参数进行【强行二次覆盖】，
        从而破解 PyTorch 会用旧参数覆盖代码修改的限制，使新参数在断点续训时立即生效。
        """
        # 1. 先让基类把历史状态（包括旧的运行轮次、旧的 LR）读取进来
        super().load_checkpoint(filename_or_checkpoint)
        
        # 2. 强行更新目标总轮数与步数，保证过渡安全
        self.num_epochs = self.custom_num_epochs
        self.num_iterations_per_epoch = self.custom_num_iterations_per_epoch
        
        # 3. 【核心注入】强行重载优化器内部参数组，让新设定的初始 LR 和权重衰减绑定
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['initial_lr'] = self.custom_initial_lr
                param_group['weight_decay'] = self.custom_weight_decay
            
            # 根据当前恢复的轮次，利用 Poly 衰减公式重新校准当前的真实运行学习率
            current_lr = self.custom_initial_lr * ((1 - self.current_epoch / self.num_epochs) ** 0.9)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
                
            self.print_to_log_file(
                f"[参数强注] 成功从断点恢复！已强行重置基准 LR 为 {self.custom_initial_lr:.2e}，"
                f"基于当前第 {self.current_epoch} 轮计算出的实际运行 LR 为: {current_lr:.2e}"
            )

    def train_step(self, batch: dict) -> dict:
            self.optimizer.zero_grad(set_to_none=True)
            
            data = batch['data'].to(self.device, non_blocking=True)
            if isinstance(batch['target'], (list, tuple)):
                target = [i.to(self.device, non_blocking=True) for i in batch['target']]
            else:
                target = batch['target'].to(self.device, non_blocking=True)

            output = self.network(data)
            del data
            
            # 【数值防御】如果输出出现了极其微小的数值倾斜，通过如下截断强行避免无穷大
            if isinstance(output, (list, tuple)):
                output = [torch.clamp(o, min=-50.0, max=50.0) for o in output]
            else:
                output = torch.clamp(output, min=-50.0, max=50.0)

            l = self.loss(output, target)

            # 如果计算出来的 Loss 本身不幸变成了 NaN，及时捕获跳过，防止污染整个网络权重
            if torch.isnan(l):
                self.print_to_log_file("⚠️ 警告: 检测到当前 Batch 产生 NaN Loss，已自动跳过该步反向传播！")
                return {'loss': 0.0}

            l.backward()
            
            if self.custom_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.custom_max_grad_norm)
                
            self.optimizer.step()
            return {'loss': l.detach().cpu().item()}

    def run_training(self):
        """
        完整重写训练大循环：
        保持与原生 nnUNet 严格一致的生命周期钩子（以便正常触发日志、EMA 更新和保存 best 权重），
        同时在 Rank 0 嵌入逼真的 tqdm 进度条。
        """
        self.on_train_start()

        # 仅在主进程或非 DDP 模式下显示进度条，防止多卡刷屏
        is_main_process = (not self.is_ddp) or (dist.get_rank() == 0)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start() # 内部包含 network.train() 和 lr_scheduler.step()

            train_outputs = []

            # 训练阶段进度条
            if is_main_process:
                pbar = tqdm(range(self.num_iterations_per_epoch),
                            desc=f"Train Epoch {epoch} (FP32)",
                            dynamic_ncols=True)
            else:
                pbar = range(self.num_iterations_per_epoch)

            for _ in pbar:
                batch = next(self.dataloader_train)
                out = self.train_step(batch)
                train_outputs.append(out)

                if is_main_process and isinstance(out, dict) and "loss" in out:
                    loss_val = out['loss']
                    loss_str = f"{loss_val:.4f}" if not np.isnan(loss_val) else "NaN! 💥"
                    pbar.set_postfix({
                        "loss": loss_str,
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })

            self.on_train_epoch_end(train_outputs)

            # 验证阶段
            with torch.no_grad():
                self.on_validation_epoch_start() # 内部包含 network.eval()
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
            
            # 原生善后处理：内部包含打印 train_loss/val_loss、更新进度图、保存 checkpoint_latest/best 逻辑
            self.on_epoch_end()

        self.on_train_end()

    def _do_i_compile(self):
        # SegMamba 带有特化的选择性扫描 Triton/CUDA 算子，强制不启用 torch.compile，防止编译报错
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
        if hasattr(mod, "decoder") and hasattr(mod.decoder, "deep_supervision"):
            mod.decoder.deep_supervision = enabled
        if hasattr(mod, "deep_supervision"):
            mod.deep_supervision = enabled

    def _build_loss(self):
        """
        保持原生的 Loss 构造逻辑，完美支持 Region 或者是普通的 分类分割目标
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