"""
ESM-2 Guidance and Reranking for DFlowNovo.

Supports:
1. In-loop reverse CTMC logit guidance (all steps or late-stage annealed).
2. End-of-trajectory candidate sequence reranking.
3. Bidirectional token mapping between DFlowNovo vocabulary and ESM-2 vocabulary.
4. PTM-aware parent residue mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForMaskedLM

from data.constants import STANDARD_AMINO_ACIDS, PTM_PARENT_MAP
from data.data import build_vocabulary


class ESMGuidanceHelper:
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: torch.device | str = "cuda",
        vocabulary: dict[str, int] | None = None,
        fp16: bool = True,
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        self.vocab = vocabulary if vocabulary is not None else build_vocabulary()
        inv_dfm = {v: k for k, v in self.vocab.items()}
        esm_vocab = self.tokenizer.get_vocab()

        self.pad_token_id = self.vocab["<pad>"]
        self.mask_token_id = self.vocab.get("<mask_token>", self.vocab.get("<mask" + ">"))

        # 1. Map DFM token IDs to ESM token IDs for inputs
        dfm_to_esm = torch.zeros(len(self.vocab), dtype=torch.long)
        for i in range(len(self.vocab)):
            tok = inv_dfm[i]
            if tok in esm_vocab:
                dfm_to_esm[i] = esm_vocab[tok]
            elif tok in PTM_PARENT_MAP and PTM_PARENT_MAP[tok] in esm_vocab:
                dfm_to_esm[i] = esm_vocab[PTM_PARENT_MAP[tok]]
            elif tok == "<pad>":
                dfm_to_esm[i] = esm_vocab["<pad>"]
            elif tok in ("<mask_token>", "<mask" + ">"):
                dfm_to_esm[i] = esm_vocab.get("<mask_token>", esm_vocab.get("<mask" + ">"))
            else:
                dfm_to_esm[i] = esm_vocab.get("<unk>", 3)

        self.dfm_to_esm_input = dfm_to_esm.to(self.device)

        # 2. Map ESM logits to DFM decoder output classes (typically 30 classes)
        num_classes = 30
        esm_to_dfm = torch.zeros(num_classes, dtype=torch.long)
        for i in range(num_classes):
            tok = inv_dfm[i]
            if tok in esm_vocab:
                esm_to_dfm[i] = esm_vocab[tok]
            elif tok in PTM_PARENT_MAP and PTM_PARENT_MAP[tok] in esm_vocab:
                esm_to_dfm[i] = esm_vocab[PTM_PARENT_MAP[tok]]
            else:
                esm_to_dfm[i] = esm_vocab.get("<unk>", 3)

        self.esm_to_dfm_output = esm_to_dfm.to(self.device)
        self.cls_id = esm_vocab["<cls>"]
        self.eos_id = esm_vocab["<eos>"]
        self.esm_pad_id = esm_vocab["<pad>"]
        self.unk_id = esm_vocab.get("<unk>", 3)

    @torch.no_grad()
    def compute_guided_logits(
        self,
        x_t: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ESM-2 unnormalized logits projected into DFlowNovo vocabulary space.

        Args:
            x_t: Current noisy sequence of token IDs, shape (B, L).
            active_mask: Boolean tensor of active (non-padded) positions, shape (B, L).

        Returns:
            Projected ESM logits of shape (B, L, 30), on the same device as x_t.
        """
        B, L = x_t.shape
        x_esm = self.dfm_to_esm_input[x_t.clamp(0, len(self.dfm_to_esm_input) - 1)]

        # Prepend <cls> and append <eos>
        cls_tokens = torch.full((B, 1), self.cls_id, dtype=torch.long, device=self.device)
        eos_tokens = torch.full((B, 1), self.eos_id, dtype=torch.long, device=self.device)
        esm_inputs = torch.cat([cls_tokens, x_esm, eos_tokens], dim=1)

        # Construct attention mask: 1 for <cls>, active positions, and <eos>
        ones_1 = torch.ones((B, 1), dtype=torch.long, device=self.device)
        attn_mask = torch.cat([ones_1, active_mask.long(), ones_1], dim=1)

        # Forward pass through ESM-2
        out = self.model(input_ids=esm_inputs, attention_mask=attn_mask)
        esm_logits = out.logits[:, 1 : L + 1, :]  # (B, L, 33)

        # Gather to DFlowNovo classes
        projected = esm_logits[:, :, self.esm_to_dfm_output].clone()  # (B, L, 30)

        # Terminal mods (classes 27, 28, 29) are not represented in ESM-2; zero their guidance
        if projected.shape[-1] > 27:
            projected[:, :, 27:30] = 0.0

        return projected

    @torch.no_grad()
    def compute_sequence_log_likelihood(
        self,
        x_clean: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean per-residue log-likelihood under ESM-2 for candidate sequences.

        Args:
            x_clean: Resolved clean sequences (B_total, L).
            active_mask: Boolean tensor of active positions (B_total, L).

        Returns:
            Tensor of mean log-probabilities per sequence of shape (B_total,).
        """
        B, L = x_clean.shape
        x_esm = self.dfm_to_esm_input[x_clean.clamp(0, len(self.dfm_to_esm_input) - 1)]

        cls_tokens = torch.full((B, 1), self.cls_id, dtype=torch.long, device=self.device)
        eos_tokens = torch.full((B, 1), self.eos_id, dtype=torch.long, device=self.device)
        esm_inputs = torch.cat([cls_tokens, x_esm, eos_tokens], dim=1)

        ones_1 = torch.ones((B, 1), dtype=torch.long, device=self.device)
        attn_mask = torch.cat([ones_1, active_mask.long(), ones_1], dim=1)

        out = self.model(input_ids=esm_inputs, attention_mask=attn_mask)
        esm_logits = out.logits[:, 1 : L + 1, :]  # (B, L, 33)
        log_probs = F.log_softmax(esm_logits, dim=-1)

        # Gather log-probability of each residue under ESM-2
        target_tokens = x_esm.unsqueeze(-1)  # (B, L, 1)
        token_log_probs = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)  # (B, L)

        # Only score actual amino acid residues (ignore padding and unknown/terminal modifications)
        valid_mask = active_mask & (x_esm != self.unk_id)
        token_log_probs = token_log_probs * valid_mask.float()

        eff_lens = valid_mask.sum(dim=-1).float().clamp(min=1.0)
        mean_esm_log_prob = token_log_probs.sum(dim=-1) / eff_lens  # (B,)
        return mean_esm_log_prob
