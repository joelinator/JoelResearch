import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, device="cpu"):
        super().__init__()
        half_dim = embedding_dim // 2
        freqs = math.log(10_000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        embeddings = self.freqs.reshape((1,) * len(x.shape) + self.freqs.shape)
        embeddings = x.unsqueeze(-1) * embeddings
        return torch.cat((embeddings.cos(), embeddings.sin()), dim=-1)


class MzArrayEncoder(nn.Module):
    def __init__(self, lambda_min=1e-3, lambda_max=1e4, emb_dim=128, device="cpu"):
        super().__init__()
        half_dim = emb_dim // 2
        freqs = math.log(lambda_max / lambda_min) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freqs)
        freqs = freqs * 2 * torch.pi / lambda_min
        self.register_buffer("freqs", freqs)

    def forward(self, m):
        embeddings = self.freqs.reshape((1,) * len(m.shape) + self.freqs.shape)
        embeddings = m.unsqueeze(-1) * embeddings
        return torch.cat((embeddings.cos(), -embeddings.sin()), dim=-1)


class AdaNLayers(nn.Module):
    def __init__(self, emb_dim=128, n=1):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, emb_dim * n * 2))
        self.n = n

    def forward(self, conditioner):
        x = self.mlp(conditioner)
        return x.chunk(2 * self.n, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate normalized features with scale and shift: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero-Initialization for Diffusion Transformers (DiT).
    Predicts (scale, shift, gate) modulation parameters for transformer sublayers.
    Initializes the projection to zeros so blocks start as identity functions.
    """
    def __init__(self, emb_dim: int = 512, num_params: int = 9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, num_params * emb_dim),
        )
        # AdaLN-Zero: zero out the final linear layer
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)
        self.num_params = num_params

    def forward(self, conditioner: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.mlp(conditioner).chunk(self.num_params, dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.w_down(F.silu(gate) * up))


SwiGLUFFN = SwiGLU
