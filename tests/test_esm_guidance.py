import pytest
import torch

from data.data import build_vocabulary
from model.esm_guidance import ESMGuidanceHelper
from inference.predict import predict_peptide
from train.factory import build_models


@pytest.fixture(scope="module")
def esm_helper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = build_vocabulary(include_ptms=True)
    helper = ESMGuidanceHelper(
        model_name="facebook/esm2_t6_8M_UR50D",
        device=device,
        vocabulary=vocab,
        fp16=(device == "cuda"),
    )
    return helper, vocab, device


def test_esm_vocab_mapping(esm_helper):
    helper, vocab, device = esm_helper
    # Verify standard AA mapping
    assert helper.dfm_to_esm_input[vocab["A"]].item() == helper.tokenizer.get_vocab()["A"]
    assert helper.dfm_to_esm_input[vocab["K"]].item() == helper.tokenizer.get_vocab()["K"]
    assert helper.dfm_to_esm_input[vocab["R"]].item() == helper.tokenizer.get_vocab()["R"]

    # Verify PTM parent mapping
    assert helper.dfm_to_esm_input[vocab["M(ox)"]].item() == helper.tokenizer.get_vocab()["M"]
    assert helper.dfm_to_esm_input[vocab["C(cam)"]].item() == helper.tokenizer.get_vocab()["C"]

    # Verify output projection classes (30 classes)
    assert helper.esm_to_dfm_output.shape[0] == 30


def test_esm_guided_logits_shape(esm_helper):
    helper, vocab, device = esm_helper
    B, L = 2, 12
    mask_id = vocab["<mask" + ">"]
    x_t = torch.full((B, L), mask_id, dtype=torch.long, device=device)
    active_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    logits = helper.compute_guided_logits(x_t, active_mask)
    assert logits.shape == (B, L, 30)
    assert logits.dtype in (torch.float32, torch.float16)

    # Verify terminal mod classes (27, 28, 29) are zeroed
    assert torch.all(logits[:, :, 27:30] == 0.0)


def test_esm_sequence_log_likelihood(esm_helper):
    helper, vocab, device = esm_helper
    # Construct a real tryptic peptide: PEPTIDER
    pep = "PEPTIDER"
    tokens = [vocab[c] for c in pep]
    x_clean = torch.tensor([tokens], dtype=torch.long, device=device)
    active_mask = torch.ones_like(x_clean, dtype=torch.bool)

    ll = helper.compute_sequence_log_likelihood(x_clean, active_mask)
    assert ll.shape == (1,)
    assert not torch.isnan(ll).any()
    assert ll.item() < 0.0  # Log-likelihood is negative


def test_predict_peptide_with_esm_modes(esm_helper):
    helper, vocab, device = esm_helper
    spectrum_encoder, length_predictor, decoder, guidance = build_models(vocab, device)
    spectrum_encoder.eval()
    length_predictor.eval()
    decoder.eval()
    guidance.eval()

    B = 2
    mz_array = torch.linspace(100, 1500, 50, device=device).unsqueeze(0).expand(B, -1)
    intensity_array = torch.ones((B, 50), device=device)
    precursor_mass = torch.tensor([1000.0, 1200.0], device=device)
    precursor_charge = torch.tensor([2, 2], device=device)
    spectrum_mask = torch.ones((B, 50), dtype=torch.bool, device=device)

    for mode in ["none", "in_loop", "late_stage", "rerank"]:
        token_ids, pred_lengths, pred_seqs, pred_scores = predict_peptide(
            mz_array=mz_array,
            mz_complementary=mz_array,
            intensity_array=intensity_array,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            spectrum_mask=spectrum_mask,
            vocabulary=vocab,
            spectrum_encoder=spectrum_encoder,
            length_predictor=length_predictor,
            decoder=decoder,
            guidance=guidance,
            scheduler="cosine",
            num_steps=3,
            top_k_lengths=1,
            esm_guidance_helper=helper if mode != "none" else None,
            esm_mode=mode,
            esm_guidance_scale=0.2,
            return_scores=True,
        )
        assert len(pred_seqs) == B
        assert len(pred_scores) == B
        assert not torch.isnan(pred_scores).any()
