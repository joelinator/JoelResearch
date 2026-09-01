import torch

from data.lengths import length_to_active_mask
from eval.evaluate import evaluate_teacher_forced
from flow_matching.sampling import sample_noising_step_uniform
from train.io import copy_history
from train.loss import (
    gamma_schedule,
    lambda_schedule,
    length_loss,
    loss_weights,
    mass_loss_hubert,
    peptide_loss,
)


def _move_batch_to_device(batch, device):
    return tuple(
        tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
        for tensor in batch
    )


def _run_epoch(
    loader,
    vocabulary,
    guidance,
    spectrum_encoder,
    length_predictor,
    decoder,
    aa_masses,
    scheduler,
    device=None,
    optimizer=None,
    train=True,
    epoch=0,
    total_epochs=1,
):
    if train:
        spectrum_encoder.train()
        length_predictor.train()
        decoder.train()
        guidance.train()
    else:
        spectrum_encoder.eval()
        length_predictor.eval()
        decoder.eval()
        guidance.eval()

    total_samples = 0
    totals = {
        "loss": 0.0,
        "decoder_loss": 0.0,
        "length_loss": 0.0,
        "mass_loss": 0.0,
    }
    proxy_totals = {
        "token_accuracy": 0.0,
        "length_accuracy": 0.0,
        "exact_peptide_accuracy": 0.0,
    }

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for (
            mz_array,
            intensity_array,
            precursor_mass,
            precursor_charge,
            sequence,
            mz_complementary,
            length,
            padded_mask,
            spectrum_mask,
        ) in loader:
            if device is not None:
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
                ) = _move_batch_to_device(
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
                    ),
                    device,
                )

            batch_size = mz_array.shape[0]
            active_mask = length_to_active_mask(length, sequence.shape[1])
            time = torch.rand(batch_size, device=sequence.device).clamp(1e-7, 1-1e-7)
            kt, _kt_derivative = scheduler(time)

            spectrum_emb_cls, spectrum_emb_peaks, full_mask = spectrum_encoder(
                mz_array,
                mz_complementary,
                intensity_array,
                spectrum_mask,
            )

            conditioner = guidance(
                spectrum_emb_peaks,
                guidance_prob=0.1,
                need_guidance=train,
            )

            x_t = sample_noising_step_uniform(
                kt, sequence, vocabulary, padding_mask=padded_mask
            )
            length_logits = length_predictor(
                spectrum_emb_cls,
                precursor_mass,
                precursor_charge,
            )
            peptide_logits = decoder(
                time,
                precursor_mass,
                precursor_charge,
                conditioner,
                x_t,
                length,
                full_mask,
            )

            decoder_loss = peptide_loss(
                peptide_logits,
                sequence,
                pad_id=vocabulary["<pad>"],
            )
            len_loss = length_loss(length_logits, length)
            mass_loss = mass_loss_hubert(
                peptide_logits,
                aa_masses,
                precursor_mass,
                active_mask=active_mask,
            )
            lambd = lambda_schedule(epoch, total_epochs)
            gamma = gamma_schedule(epoch, total_epochs)

            loss = decoder_loss + lambd * len_loss + gamma * mass_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                all_params = (
                    list(spectrum_encoder.parameters())
                    + list(length_predictor.parameters())
                    + list(decoder.parameters())
                    + list(guidance.parameters())
                )
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

            total_samples += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["decoder_loss"] += decoder_loss.item() * batch_size
            totals["length_loss"] += len_loss.item() * batch_size
            totals["mass_loss"] += mass_loss.item() * batch_size

            if not train:
                proxy = evaluate_teacher_forced(
                    peptide_logits,
                    sequence,
                    length_logits,
                    length,
                    active_mask,
                    vocabulary,
                )
                for key in proxy_totals:
                    proxy_totals[key] += proxy[key] * batch_size

    if total_samples == 0:
        metrics = {key: 0.0 for key in totals}
        metrics.update({key: 0.0 for key in proxy_totals})
        return metrics

    metrics = {key: value / total_samples for key, value in totals.items()}
    if not train:
        metrics.update({key: value / total_samples for key, value in proxy_totals.items()})
    return metrics


def training_loop(
    optimizer,
    epochs,
    vocabulary,
    guidance,
    spectrum_encoder,
    length_predictor,
    decoder,
    aa_masses,
    train_loader,
    valid_loader,
    scheduler,
    device=None,
    total_epochs=None,
    initial_history=None,
    epoch_end_callback=None,
    eval_every: int = 1,
    eval_max_batches: int = 32,
    inference_steps: int = 50,
    noising_scheme: str = "uniform",
    guidance_scale: float = 1.0,
):
    from eval.evaluate import evaluate_generative
    from eval.metrics import format_metrics

    history = copy_history(initial_history)
    for key in (
        "valid_token_accuracy",
        "valid_length_accuracy",
        "valid_exact_peptide_accuracy",
        "valid_aa_precision",
        "valid_aa_recall",
        "valid_peptide_precision",
        "valid_peptide_recall",
        "valid_exact_generative_accuracy",
    ):
        history.setdefault(key, [])
    if total_epochs is None:
        epoch_list = list(epochs)
        total_epochs = epoch_list[-1] + 1 if epoch_list else 0

    for epoch in epochs:
        weights = loss_weights(epoch, total_epochs)
        history["lambda"].append(weights["lambda"])
        history["gamma"].append(weights["gamma"])
        train_metrics = _run_epoch(
            train_loader,
            vocabulary,
            guidance,
            spectrum_encoder,
            length_predictor,
            decoder,
            aa_masses,
            scheduler,
            device=device,
            optimizer=optimizer,
            train=True,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        valid_metrics = _run_epoch(
            valid_loader,
            vocabulary,
            guidance,
            spectrum_encoder,
            length_predictor,
            decoder,
            aa_masses,
            scheduler,
            device=device,
            train=False,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_decoder_loss"].append(train_metrics["decoder_loss"])
        history["train_length_loss"].append(train_metrics["length_loss"])
        history["train_mass_loss"].append(train_metrics["mass_loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["valid_decoder_loss"].append(valid_metrics["decoder_loss"])
        history["valid_length_loss"].append(valid_metrics["length_loss"])
        history["valid_mass_loss"].append(valid_metrics["mass_loss"])
        history["valid_token_accuracy"].append(valid_metrics.get("token_accuracy", 0.0))
        history["valid_length_accuracy"].append(valid_metrics.get("length_accuracy", 0.0))
        history["valid_exact_peptide_accuracy"].append(
            valid_metrics.get("exact_peptide_accuracy", 0.0)
        )

        generative_metrics = None
        if eval_max_batches > 0 and (epoch % eval_every == 0 or epoch == total_epochs - 1):
            generative_metrics = evaluate_generative(
                valid_loader,
                vocabulary,
                spectrum_encoder,
                length_predictor,
                decoder,
                guidance,
                scheduler,
                device,
                max_batches=eval_max_batches,
                num_steps=inference_steps,
                noising_scheme=noising_scheme,
                guidance_scale=guidance_scale,
            )
            history["valid_aa_precision"].append(generative_metrics.aa_precision)
            history["valid_aa_recall"].append(generative_metrics.aa_recall)
            history["valid_peptide_precision"].append(generative_metrics.peptide_precision)
            history["valid_peptide_recall"].append(generative_metrics.peptide_recall)
            history["valid_exact_generative_accuracy"].append(
                generative_metrics.exact_peptide_accuracy
            )

        print(
            f"epoch {epoch + 1}/{total_epochs} "
            f"λ={weights['lambda']:.3f} γ={weights['gamma']:.3f} "
            f"train={train_metrics['loss']:.4f} valid={valid_metrics['loss']:.4f} "
            f"tf_tok={valid_metrics.get('token_accuracy', 0.0):.4f} "
            f"tf_len={valid_metrics.get('length_accuracy', 0.0):.4f}"
        )
        if generative_metrics is not None:
            print(f"  generative: {format_metrics(generative_metrics)}")
        if epoch_end_callback is not None:
            epoch_end_callback(
                epoch=epoch,
                total_epochs=total_epochs,
                history=history,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                weights=weights,
                generative_metrics=generative_metrics.to_dict() if generative_metrics else None,
            )

    return history


# Backward-compatible alias.
trainning_loop = training_loop
