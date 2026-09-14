"""PyTorch Lightning Callbacks for DFM Training."""

from __future__ import annotations

import pytorch_lightning as pl
import torch


class EMACallback(pl.Callback):
    """
    Exponential Moving Average (EMA) callback for model parameters.

    Maintains a shadow copy of trainable model parameters updated after each training step:
        shadow = decay * shadow + (1 - decay) * current_weight

    During validation and testing, swaps model parameters with EMA shadow weights
    so that generative evaluation metrics reflect the smoothed weights. Restores
    standard training parameters upon completion of validation/testing.

    Also persists and restores EMA shadow weights across checkpoint saving/loading.
    """

    def __init__(self, decay: float = 0.999, update_interval: int = 1):
        super().__init__()
        self.decay = decay
        self.update_interval = update_interval
        self.shadow_params: dict[str, torch.Tensor] = {}
        self.backup_params: dict[str, torch.Tensor] = {}

    def _init_shadow_params(self, pl_module: pl.LightningModule) -> None:
        self.shadow_params = {
            name: param.detach().clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.shadow_params:
            self._init_shadow_params(pl_module)
        else:
            # Ensure device alignment if loaded from checkpoint
            device = pl_module.device
            self.shadow_params = {
                name: tensor.to(device) for name, tensor in self.shadow_params.items()
            }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if (trainer.global_step + 1) % self.update_interval == 0:
            if not self.shadow_params:
                self._init_shadow_params(pl_module)
            with torch.no_grad():
                one_minus_decay = 1.0 - self.decay
                for name, param in pl_module.named_parameters():
                    if param.requires_grad and name in self.shadow_params:
                        self.shadow_params[name].mul_(self.decay).add_(
                            param.data, alpha=one_minus_decay
                        )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Swap in EMA weights for validation evaluation
        if self.shadow_params:
            self.backup_params = {
                name: param.detach().clone()
                for name, param in pl_module.named_parameters()
                if param.requires_grad
            }
            with torch.no_grad():
                for name, param in pl_module.named_parameters():
                    if name in self.shadow_params:
                        param.data.copy_(self.shadow_params[name])

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Restore trained weights after validation
        if self.backup_params:
            with torch.no_grad():
                for name, param in pl_module.named_parameters():
                    if name in self.backup_params:
                        param.data.copy_(self.backup_params[name])
            self.backup_params.clear()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.shadow_params:
            self.backup_params = {
                name: param.detach().clone()
                for name, param in pl_module.named_parameters()
                if param.requires_grad
            }
            with torch.no_grad():
                for name, param in pl_module.named_parameters():
                    if name in self.shadow_params:
                        param.data.copy_(self.shadow_params[name])

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.backup_params:
            with torch.no_grad():
                for name, param in pl_module.named_parameters():
                    if name in self.backup_params:
                        param.data.copy_(self.backup_params[name])
            self.backup_params.clear()

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> None:
        if self.shadow_params:
            checkpoint["ema_shadow_params"] = {
                k: v.detach().cpu() for k, v in self.shadow_params.items()
            }

    def on_load_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> None:
        if "ema_shadow_params" in checkpoint:
            self.shadow_params = {
                k: v.clone() for k, v in checkpoint["ema_shadow_params"].items()
            }
