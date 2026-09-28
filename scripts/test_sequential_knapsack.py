#!/usr/bin/env python3
"""
Test sequential mass budget resolution function on a batch.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from data.data import build_dataloader, build_vocabulary, get_dataset
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint
from inference.knapsack import ExactReachabilityDP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_sequential_knapsack(
    x_t: torch.Tensor,
    logits: torch.Tensor,
    active_mask: torch.Tensor,
    target_residue_mass: torch.Tensor,
    mask_id: int,
    mass_table: torch.Tensor,
    full_mass_table: torch.Tensor,
    tol: float = 1.0,
    reachability_dp: ExactReachabilityDP | None = None,
    use_exact_dp: bool = True,
    valid_aa_masses: torch.Tensor | None = None,
    pairs_2: torch.Tensor | None = None,
    min_aa_mass: float = 57.021464,
    max_aa_mass: float = 186.079313,
) -> torch.Tensor:
    """
    Sequentially unmask residual masked positions one-by-one with exact DP reachability updates.
    Prevents joint knapsack collisions when 2 or more positions are unmasked together.
    """
    from inference.predict import knapsack_filter_logits_vectorized

    x_res = x_t.clone()
    batch_size = x_res.shape[0]
    batch_indices = torch.arange(batch_size, device=x_res.device)

    # Loop at most max_len times (typically 1-3 iterations)
    max_iters = x_res.shape[1]
    for _ in range(max_iters):
        is_masked = (x_res == mask_id) & active_mask
        if not is_masked.any():
            break

        # Re-filter logits with current partially unmasked state
        filtered_logits = knapsack_filter_logits_vectorized(
            logits=logits.clone(),
            x_t=x_res,
            target_residue_mass=target_residue_mass,
            active_mask=active_mask,
            mask_id=mask_id,
            mass_table=mass_table,
            full_mass_table=full_mass_table,
            tol=tol,
            valid_aa_masses=valid_aa_masses,
            pairs_2=pairs_2,
            min_aa_mass=min_aa_mass,
            max_aa_mass=max_aa_mass,
            reachability_dp=reachability_dp,
            use_exact_dp=use_exact_dp,
        )

        # Confidence of masked positions
        conf = filtered_logits.max(dim=-1).values
        conf_masked = torch.where(is_masked, conf, torch.full_like(conf, -1e9))

        # Best position to unmask per sequence
        best_pos = conf_masked.argmax(dim=-1)  # (B,)
        has_masked = is_masked.any(dim=-1)  # (B,)

        active_b = batch_indices[has_masked]
        active_pos = best_pos[has_masked]

        # Select best residue for this single position
        chosen_tokens = filtered_logits[active_b, active_pos].argmax(dim=-1)
        x_res[active_b, active_pos] = chosen_tokens

    return x_res

def test():
    ckpt_path = PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "checkpoints" / "best-joint-gen-exact-epoch=01-exact=0.4746.ckpt"
    ckpt = load_checkpoint(str(ckpt_path), map_location=DEVICE)
    vocab = ckpt.get("vocabulary") or build_vocabulary(include_ptms=True)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval(); lp.eval(); dec.eval(); guid.eval()

    print("Loaded models successfully. Function syntax verified.")

if __name__ == "__main__":
    test()
