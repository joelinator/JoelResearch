"""PyTorch Lightning module for DFM Multi-GPU training."""

from __future__ import annotations

import torch
import pytorch_lightning as pl

from data.data import get_output_aa_masses
from data.lengths import length_to_active_mask
from eval.evaluate import evaluate_teacher_forced
from flow_matching.sampling import (
    sample_noising_step_mask,
    sample_noising_step_uniform,
)
from flow_matching.scheduler import cosine_scheduler
from train.factory import build_models
from train.loss import (
    gamma_schedule,
    lambda_schedule,
    length_loss,
    mass_loss_hubert,
    peptide_loss,
)


class DFMLightningModule(pl.LightningModule):
    def __init__(
        self,
        vocabulary: dict[str, int],
        args: dict,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vocabulary"])
        self.vocabulary = vocabulary
        self.args = args

        compile_models = self.args.get("compile", False)
        (
            self.spectrum_encoder,
            self.length_predictor,
            self.decoder,
            self.guidance,
        ) = build_models(
            vocabulary,
            device=torch.device("cpu"),
            compile_models=compile_models,
        )

        aa_masses = get_output_aa_masses(vocabulary)
        self.register_buffer("aa_masses", aa_masses)

        self.scheduler = cosine_scheduler

    def _shared_step(self, batch, batch_idx, mode="train"):
        (
            mz_array,
            intensity_array,
            precursor_mass,
            precursor_charge,
            sequence,
            mz_complementary,
            length,
            padded_mask,
            spectrum_mask,
        ) = batch

        batch_size = mz_array.shape[0]
        active_mask = length_to_active_mask(length, sequence.shape[1])
        time = torch.rand(batch_size, device=self.device).clamp(1e-7, 1 - 1e-7)
        kt, _kt_derivative = self.scheduler(time)

        spectrum_emb_cls, spectrum_emb_peaks, peak_mask = self.spectrum_encoder(
            mz_array,
            mz_complementary,
            intensity_array,
            spectrum_mask,
        )

        conditioner = self.guidance(
            spectrum_emb_peaks,
            guidance_prob=0.1 if mode == "train" else 0.0,
            need_guidance=(mode == "train"),
        )

        noising_scheme = self.args.get("noising_scheme", "mask")
        if noising_scheme == "mask":
            x_t = sample_noising_step_mask(kt, sequence, self.vocabulary, padding_mask=padded_mask)
        else:
            x_t = sample_noising_step_uniform(kt, sequence, self.vocabulary, padding_mask=padded_mask)

        length_logits = self.length_predictor(
            spectrum_emb_cls,
            precursor_mass,
            precursor_charge,
        )
        peptide_logits = self.decoder(
            time,
            precursor_mass,
            precursor_charge,
            conditioner,
            x_t,
            length.float(),
            peak_mask,
        )

        decoder_loss = peptide_loss(
            peptide_logits,
            sequence,
            pad_id=self.vocabulary["<pad>"],
        )
        len_loss = length_loss(length_logits, length)
        mass_loss = mass_loss_hubert(
            peptide_logits,
            self.aa_masses,
            precursor_mass,
            active_mask=active_mask,
        )

        epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs or self.args.get("epochs", 30)
        lambd = lambda_schedule(epoch, total_epochs)
        gamma = gamma_schedule(epoch, total_epochs)

        loss = decoder_loss + lambd * len_loss + gamma * mass_loss

        self.log(f"{mode}/loss", loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{mode}/decoder_loss", decoder_loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{mode}/length_loss", len_loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{mode}/mass_loss", mass_loss, batch_size=batch_size, sync_dist=True)

        if mode == "valid":
            proxy = evaluate_teacher_forced(
                peptide_logits,
                sequence,
                length_logits,
                length,
                active_mask,
                self.vocabulary,
            )
            for k, v in proxy.items():
                self.log(f"valid/{k}", v, batch_size=batch_size, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="valid")

    def configure_optimizers(self):
        import math
        from torch.optim.lr_scheduler import LambdaLR

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.get("learning_rate", 3e-4),
            weight_decay=self.args.get("weight_decay", 0.01),
        )
        
        # PyTorch Lightning helper to get the exact total number of optimization steps
        # across all epochs, accounting for multi-GPU and gradient accumulation.
        total_steps = self.trainer.estimated_stepping_batches
        
        # Warmup for 5% of total training steps
        warmup_steps = int(total_steps * 0.05)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Crucial: update LR every step, not every epoch!
            },
        }
