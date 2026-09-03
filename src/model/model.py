import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    AdaNLayers,
    AdaLNZero,
    MzArrayEncoder,
    SinusoidalEmbedding,
    SwiGLU,
    SwiGLUFFN,
    modulate,
)
from data.lengths import MAX_PEPTIDE_LENGTH, MIN_PEPTIDE_LENGTH, NUM_LENGTH_CLASSES


class SpectrumEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mz_encoder = MzArrayEncoder(emb_dim=model_dim // 2)
        self.comp_mz_encoder = MzArrayEncoder(emb_dim=model_dim // 2)
        self.intensity_proj = nn.Linear(1, model_dim)
        self.peak_proj = nn.Linear(model_dim * 2, model_dim)
        self.peak_norm = nn.LayerNorm(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm avoids vanishing gradients in deep layers
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(model_dim)
        self.cls_token = nn.Parameter(torch.randn(model_dim))

    def forward(self, mz1, mz2, intensity, spectrum_mask):
        """
        Args:
            mz1: mass array (B, S)
            mz2: complementary mass array (B, S)
            intensity: peak intensities (B, S)
            spectrum_mask: padding mask where True marks padded positions (B, S)

        Returns:
            cls_embedding (B, D), peak_embeddings (B, S, D), peak_mask (B, S)
        """
        batch_size = mz1.shape[0]
        device = mz1.device

        cls_token = self.cls_token.expand(batch_size, 1, -1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat((cls_mask, spectrum_mask), dim=1)

        # Fuse mz1, complementary mz2, and intensity per peak into a single D-dim token
        mz1_emb = self.mz_encoder(mz1)
        mz2_emb = self.comp_mz_encoder(mz2)
        mz_combined = torch.cat([mz1_emb, mz2_emb], dim=-1)
        int_emb = self.intensity_proj(intensity.unsqueeze(-1))
        peak_tokens = self.peak_norm(self.peak_proj(torch.cat([mz_combined, int_emb], dim=-1)))

        x = torch.cat((cls_token, peak_tokens), dim=1)
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        x = self.final_norm(x)
        return x[:, 0, :], x[:, 1:, :], spectrum_mask


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim=512, num_heads=8, mlp_hidden_dim=1536, dropout=0.1):
        super().__init__()
        # 9 modulation parameters with zero-initialized gates (AdaLN-Zero):
        # (shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp)
        self.ada_ln = AdaLNZero(emb_dim=emb_dim, num_params=9)
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.mlp = SwiGLUFFN(d_model=emb_dim, d_ff=mlp_hidden_dim, dropout=dropout)

    def forward(self, conditioner, x, y, full_mask=None):
        (
            shift_msa, scale_msa, gate_msa,
            shift_cross, scale_cross, gate_cross,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.ada_ln(conditioner)

        # 1. Self-Attention with AdaLN modulation and residual gate
        norm_x1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.self_attention(query=norm_x1, key=norm_x1, value=norm_x1)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. Cross-Attention with AdaLN modulation and residual gate
        norm_x2 = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_out, _ = self.cross_attention(
            query=norm_x2,
            key=y,
            value=y,
            key_padding_mask=full_mask,
        )
        x = x + gate_cross.unsqueeze(1) * cross_out

        # 3. Feed-Forward SwiGLU with AdaLN modulation and residual gate
        norm_x3 = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x3)
        return x


class DFMPeptideDecoder(nn.Module):
    def __init__(
        self,
        spec_dim: int = 512,
        emb_dim: int = 512,
        mlp_hidden_dim: int = 1536,
        vocab_size=22,
        mask_token_id=None,
        pad_token_id=None,
        max_charge=6,
        min_length: int = MIN_PEPTIDE_LENGTH,
        max_length: int = MAX_PEPTIDE_LENGTH,
        n_decoder_blocks: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if mask_token_id is None:
            mask_token_id = vocab_size - 1
        if pad_token_id is None:
            pad_token_id = vocab_size - 2

        output_token_ids = [
            token_id
            for token_id in range(vocab_size)
            if token_id not in {mask_token_id, pad_token_id}
        ]
        if output_token_ids != list(range(vocab_size - 2)):
            raise ValueError(
                "Decoder output logits require <pad> and <mask_token> to be the last two "
                f"vocabulary entries (pad={pad_token_id}, mask={mask_token_id}, "
                f"vocab_size={vocab_size})."
            )

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.output_vocab_size = vocab_size - 2

        self.sinusoidal_embedding = SinusoidalEmbedding(emb_dim)
        # Input embedding uses the full vocabulary (noisy x_t may contain <mask_token>).
        self.peptide_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        # Charges are 1-indexed in the dataset; reserve index 0 for padding/unused.
        self.charge_embedding = nn.Embedding(
            num_embeddings=max_charge + 1,
            embedding_dim=emb_dim,
        )

        self.min_length = min_length
        self.max_length = max_length
        # Lengths are in [min_length, max_length]; reserve indices 0..max_length for 1-based indexing.
        self.length_embedding = nn.Embedding(
            num_embeddings=max_length + 1,
            embedding_dim=emb_dim,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_decoder_blocks)
            ]
        )

        self.spectrum_proj = nn.Linear(spec_dim, emb_dim)
        self.conditioner_proj = nn.Linear(4 * emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.ada_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim),
        )
        nn.init.zeros_(self.ada_final[1].weight)
        nn.init.zeros_(self.ada_final[1].bias)

        # Output head: Linear projection from conditioned representation to vocabulary
        self.head = nn.Linear(emb_dim, self.output_vocab_size)

    @classmethod
    def from_vocabulary(cls, vocab: dict[str, int], **kwargs):
        """Build a decoder whose output head excludes the <mask_token>."""
        mask_id = vocab.get("<mask_token>", vocab.get("<mask>"))
        if mask_id is None:
            raise KeyError("Vocabulary must contain either '<mask_token>' or '<mask>'.")
        return cls(
            vocab_size=len(vocab),
            mask_token_id=mask_id,
            pad_token_id=vocab["<pad>"],
            **kwargs,
        )

    def forward(
        self,
        time,
        precursor_m,
        precursor_c,
        spectrum_embeddings,
        peptide_seq,
        length,
        full_mask=None,
    ):
        seq_len = peptide_seq.shape[1]
        t = self.sinusoidal_embedding(time)

        peptide_emb = self.peptide_embedding(peptide_seq)
        positions = torch.arange(seq_len, device=peptide_seq.device, dtype=torch.float32)
        pos_emb = self.sinusoidal_embedding(positions).unsqueeze(0)

        charge_index = precursor_c.long().clamp(min=1, max=self.charge_embedding.num_embeddings - 1)
        c_emb = self.charge_embedding(charge_index)
        m_emb = self.sinusoidal_embedding(precursor_m)

        length_index = length.long().clamp(min=self.min_length, max=self.max_length)
        length_emb = self.length_embedding(length_index)

        if m_emb.dim() > 2:
            m_emb = m_emb.squeeze(1)
        if c_emb.dim() > 2:
            c_emb = c_emb.squeeze(1)
        if length_emb.dim() > 2:
            length_emb = length_emb.squeeze(1)
        if t.dim() > 2:
            t = t.squeeze(1)

        precursor_m_c_t_l = self.conditioner_proj(
            torch.cat((c_emb, m_emb, t, length_emb), dim=-1)
        )

        x = peptide_emb + pos_emb
        y = self.spectrum_proj(spectrum_embeddings)
        for block in self.decoder_blocks:
            x = block(precursor_m_c_t_l, x, y, full_mask)

        shift_final, scale_final = self.ada_final(precursor_m_c_t_l).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift_final, scale_final)
        # Output logits: amino acids only (no <pad>, no <mask_token>).
        return self.head(x)


class PeptideLengthClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int = NUM_LENGTH_CLASSES,
        min_length: int = MIN_PEPTIDE_LENGTH,
        max_length: int = MAX_PEPTIDE_LENGTH,
        in_dim: int = 512,
        hidden_dim: int = 256,
        emb_dim: int = 128,
        max_charge: int = 10,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self.n_classes = n_classes
        self.mass_emb = SinusoidalEmbedding(emb_dim)
        self.charge_emb = nn.Embedding(max_charge + 1, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 2 * emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_classes),
        )

    def forward(self, spectrum_embeddings, precursor_mass, precursor_charge):
        m = self.mass_emb(precursor_mass.reshape(-1))
        charge_index = precursor_charge.long().clamp(
            min=0, max=self.charge_emb.num_embeddings - 1
        )
        c = self.charge_emb(charge_index)
        x = torch.cat([spectrum_embeddings.reshape(m.shape[0], -1), m, c], dim=-1)
        return self.mlp(x)


# Backward-compatible aliases for older imports.
DecoderBloc = DecoderBlock
PeptideLenghtClassifier = PeptideLengthClassifier
