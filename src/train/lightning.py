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
from train.utils import length_noiser


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

        true_length = length
        is_length_noised = False
        use_length_noising = self.args.get("length_noising", True)
        noising_prob = self.args.get("length_noising_prob", 0.1)

        if mode == "train" and use_length_noising:
            length_for_decoder, is_length_noised = length_noiser(
                length, noising_prob=noising_prob
            )
        else:
            length_for_decoder = length

        peptide_logits = self.decoder(
            time,
            precursor_mass,
            precursor_charge,
            conditioner,
            x_t,
            length_for_decoder,
            peak_mask,
        )

        decoder_loss = peptide_loss(
            peptide_logits,
            sequence,
            pad_id=self.vocabulary["<pad>"],
        )
        len_loss = length_loss(length_logits, true_length)

        epoch = self.current_epoch
        trainer = getattr(self, "_trainer", None)
        total_epochs = (trainer.max_epochs if trainer is not None else None) or self.args.get("epochs", 30)
        lambd = lambda_schedule(epoch, total_epochs)
        gamma = gamma_schedule(epoch, total_epochs)

        if not is_length_noised:
            mass_loss = mass_loss_hubert(
                peptide_logits,
                self.aa_masses,
                precursor_mass,
                active_mask=active_mask,
            )
            loss = decoder_loss + lambd * len_loss + gamma * mass_loss
            mass_loss_log = mass_loss
        else:
            loss = decoder_loss + lambd * len_loss
            mass_loss_log = torch.tensor(0.0, device=loss.device)

        self.log(f"{mode}/loss", loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{mode}/decoder_loss", decoder_loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{mode}/length_loss", len_loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{mode}/mass_loss", mass_loss_log, batch_size=batch_size, sync_dist=True)

        if mode == "valid":
            self.log("valid_loss", loss, batch_size=batch_size, sync_dist=True)
            proxy = evaluate_teacher_forced(
                peptide_logits,
                sequence,
                length_logits,
                true_length,
                active_mask,
                self.vocabulary,
            )
            for k, v in proxy.items():
                self.log(f"valid/{k}", v, batch_size=batch_size, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="train")

    def on_validation_epoch_start(self):
        self._val_proxy_batches = []

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, mode="valid")
        max_proxy_batches = int(self.args.get("gen_eval_proxy_batches", 5))
        if max_proxy_batches > 0 and len(self._val_proxy_batches) < max_proxy_batches:
            self._val_proxy_batches.append(batch)
        return loss

    def on_validation_epoch_end(self):
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            self._val_proxy_batches = []
            return

        if not getattr(self, "_val_proxy_batches", None):
            return

        from inference.predict import predict_peptide
        from eval.evaluate import sequences_from_batch
        from eval.metrics import compute_denovo_metrics

        num_steps = int(self.args.get("inference_steps", 20))
        guidance_scale = float(self.args.get("guidance_scale", 1.5))
        top_k_lengths = int(self.args.get("top_k_lengths", 3))
        alpha = float(self.args.get("length_beam_alpha", 0.01))

        predictions = []
        targets = []
        pred_lengths_all = []
        target_lengths_all = []

        device = self.device
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self.spectrum_encoder.eval()
        self.length_predictor.eval()
        self.decoder.eval()
        self.guidance.eval()

        with torch.no_grad():
            for batch in self._val_proxy_batches:
                (
                    mz_array,
                    intensity_array,
                    precursor_mass,
                    precursor_charge,
                    sequence,
                    mz_complementary,
                    length,
                    _padded_mask,
                    spectrum_mask,
                ) = tuple(
                    t.to(device) if torch.is_tensor(t) else t for t in batch
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=(device.type == "cuda"),
                ):
                    token_ids, pred_lens, pred_seqs = predict_peptide(
                        mz_array=mz_array,
                        intensity_array=intensity_array,
                        precursor_mass=precursor_mass,
                        precursor_charge=precursor_charge,
                        mz_complementary=mz_complementary,
                        spectrum_mask=spectrum_mask,
                        vocabulary=self.vocabulary,
                        spectrum_encoder=self.spectrum_encoder,
                        length_predictor=self.length_predictor,
                        decoder=self.decoder,
                        guidance=self.guidance,
                        scheduler=self.scheduler,
                        num_steps=num_steps,
                        noising_scheme="mask",
                        guidance_scale=guidance_scale,
                        top_k_lengths=top_k_lengths,
                        alpha=alpha,
                    )

                predictions.extend(pred_seqs)
                targets.extend(sequences_from_batch(sequence, self.vocabulary))
                pred_lengths_all.extend(int(v) for v in pred_lens.tolist())
                target_lengths_all.extend(int(v) for v in length.tolist())

        self._val_proxy_batches = []

        if len(predictions) > 0:
            metrics = compute_denovo_metrics(
                predictions,
                targets,
                predicted_lengths=pred_lengths_all,
                target_lengths=target_lengths_all,
                aa_mass_tolerance=0.1,
                prefix_mass_tolerance=0.5,
            )

            # Flat names for clean checkpoint filenames
            self.log("valid_gen_exact_match", metrics.exact_peptide_accuracy, sync_dist=True)
            self.log("valid_gen_peptide_f1", metrics.peptide_f1, sync_dist=True)
            self.log("valid_gen_aa_f1", metrics.aa_f1, sync_dist=True)
            self.log("valid_gen_length_acc", metrics.length_accuracy, sync_dist=True)

            # Nested names for TensorBoard grouping
            self.log("valid/gen_exact_match", metrics.exact_peptide_accuracy, sync_dist=True)
            self.log("valid/gen_peptide_f1", metrics.peptide_f1, sync_dist=True)
            self.log("valid/gen_aa_f1", metrics.aa_f1, sync_dist=True)
            self.log("valid/gen_length_acc", metrics.length_accuracy, sync_dist=True)

            print(
                f"\n[Validation Generative Proxy ({len(predictions)} spectra)] "
                f"Exact={metrics.exact_peptide_accuracy * 100:.2f}%, "
                f"PepF1={metrics.peptide_f1 * 100:.2f}%, "
                f"AAF1={metrics.aa_f1 * 100:.2f}%, "
                f"LenAcc={metrics.length_accuracy * 100:.2f}%",
                flush=True,
            )

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

        min_lr_ratio = float(self.args.get("min_lr_ratio", 0.05))

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay with minimum learning rate floor to prevent late training freeze
            progress = min(1.0, float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Crucial: update LR every step, not every epoch!
            },
        }
