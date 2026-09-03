import torch
import torch.nn as nn


class ClfGuidance(nn.Module):
    """Classifier-free guidance dropout for spectrum conditioning."""

    def __init__(self, cond_dim=512):
        super().__init__()
        self.unconditional = nn.Parameter(torch.randn(cond_dim))

    def forward(self, batch_data, guidance_prob=0.1, need_guidance=True):
        if not need_guidance or guidance_prob <= 0.0:
            return batch_data

        device = batch_data.device
        mask = (torch.rand(batch_data.shape[0], device=device) < guidance_prob).view(-1, 1, 1)
        unconditional = self.unconditional.view(1, 1, -1).expand_as(batch_data)
        return torch.where(mask, unconditional, batch_data)
