import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AdaNLayers, MzArrayEncoder, SinusoidalEmbedding, SwiGLU
from data.lengths import MAX_PEPTIDE_LENGTH, MIN_PEPTIDE_LENGTH, NUM_LENGTH_CLASSES


class SpectrumEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mz_encoder = MzArrayEncoder(emb_dim=model_dim)
        self.intensity_weight = nn.Parameter(torch.randn(model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(model_dim))

    def forward(self, mz1, mz2, intensity, spectrum_mask):
        """
        Args:
            mz1: mass array (B, S)
            mz2: complementary mass array (B, S)
            intensity: peak intensities (B, S)
            spectrum_mask: padding mask where True marks padded positions (B, S)

        Returns:
            cls_embedding (B, D), full_sequence_embeddings (B, 3S+1, D),
            peak_mask (B, 3S) for cross-attention in the decoder
        """
        batch_size = mz1.shape[0]
        device = mz1.device

        cls_token = self.cls_token.expand(batch_size, 1, -1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat((cls_mask, spectrum_mask, spectrum_mask, spectrum_mask), dim=1)

        intensity_emb = intensity.unsqueeze(-1) * self.intensity_weight.reshape(
            (1,) * len(intensity.shape) + self.intensity_weight.shape
        )
        x = torch.cat(
            (
                cls_token,
                self.mz_encoder(mz1),
                self.mz_encoder(mz2),
                intensity_emb,
            ),
            dim=1,
        )
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        return x[:, 0, :], x[:,1:,:], full_mask[:,1:]


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim=128, num_heads=4, mlp_hidden_dim=128):
        super().__init__()
        self.ada_n_layers = AdaNLayers(emb_dim, n=3)
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            emb_dim, num_heads=num_heads, batch_first=True
        )
        self.norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(3)])
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim * 2),
            nn.Dropout(),
            SwiGLU(mlp_hidden_dim * 2, mlp_hidden_dim),
        )

    def forward(self, conditioner, x, y, full_mask=None):
        print("----->>>> DEBUG:::", full_mask.shape, x.shape, y.shape)
        params = self.ada_n_layers(conditioner)
        scales = params[:3]
        shifts = params[3:]

        x = self.norms[0](x) * (1 + scales[0]).unsqueeze(1) + shifts[0].unsqueeze(1)
        x, _ = self.self_attention(query=x, key=x, value=x)

        x = self.norms[1](x) * (1 + scales[1]).unsqueeze(1) + shifts[1].unsqueeze(1)
        x, _ = self.cross_attention(
            query=x,
            key=y,
            value=y,
            key_padding_mask=full_mask,
        )

        x = self.norms[2](x) * (1 + scales[2]).unsqueeze(1) + shifts[2].unsqueeze(1)
        return self.mlp(x)


class DFMPeptideDecoder(nn.Module):
    def __init__(
        self,
        spec_dim: int = 512,
        emb_dim: int = 512,
        mlp_hidden_dim: int = 512,
        vocab_size=22,
        mask_token_id=None,
        pad_token_id=None,
        max_charge=6,
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
                "Decoder output logits require <pad> and <mask> to be the last two "
                f"vocabulary entries (pad={pad_token_id}, mask={mask_token_id}, "
                f"vocab_size={vocab_size})."
            )

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.output_vocab_size = vocab_size - 2

        self.sinusoidal_embedding = SinusoidalEmbedding(emb_dim)
        # Input embedding uses the full vocabulary (noisy x_t may contain <mask>).
        self.peptide_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        # Charges are 1-indexed in the dataset; reserve index 0 for padding/unused.
        self.charge_embedding = nn.Embedding(
            num_embeddings=max_charge + 1,
            embedding_dim=emb_dim,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(emb_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim)
                for _ in range(n_decoder_blocks)
            ]
        )

        self.spectrum_proj = nn.Linear(spec_dim, emb_dim)
        self.conditioner_proj = nn.Linear(4 * emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.ada_n_layer = AdaNLayers(emb_dim, n=1)

        self.head = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            SwiGLU(mlp_hidden_dim, mlp_hidden_dim * 2),
            nn.Linear(mlp_hidden_dim * 2, self.output_vocab_size),
        )

    @classmethod
    def from_vocabulary(cls, vocab: dict[str, int], **kwargs):
        """Build a decoder whose output head excludes the <mask> token."""
        return cls(
            vocab_size=len(vocab),
            mask_token_id=vocab["<mask>"],
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
        length_emb = self.sinusoidal_embedding(length.float())

        peptide_emb = self.peptide_embedding(peptide_seq)
        positions = torch.arange(seq_len, device=peptide_seq.device, dtype=torch.float32)
        pos_emb = self.sinusoidal_embedding(positions).unsqueeze(0)

        charge_index = precursor_c.long().clamp(min=0, max=self.charge_embedding.num_embeddings - 1)
        c_emb = self.charge_embedding(charge_index)
        m_emb = self.sinusoidal_embedding(precursor_m)

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

        params = self.ada_n_layer(precursor_m_c_t_l)
        x = self.norm(x) * (1 + params[0]).unsqueeze(1) + params[1].unsqueeze(1)
        # Output logits: amino acids only (no <pad>, no <mask>).
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
        max_charge=10,
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
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
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
