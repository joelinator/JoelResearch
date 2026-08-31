import torch
import torch.nn as nn


class ClfGuidance(nn.Module):
    """Classifier-free guidance dropout for spectrum conditioning."""

    def __init__(self, cond_dim=128):
        super().__init__()
        self.unconditional = nn.Parameter(torch.randn(cond_dim))

    def forward(self, batch_data, guidance_prob=0.1, need_guidance=True):
        if not need_guidance:
            return batch_data

        device = batch_data.device
        mask = torch.rand(batch_data.shape[0], device=device) < guidance_prob
        mask = mask.reshape(mask.shape + (1,) * (len(batch_data.shape) - 1))
        unconditional = self.unconditional.reshape((1,) * (len(batch_data.shape) - 1) + self.unconditional.shape)
        return torch.where(mask, unconditional.expand_as(batch_data), batch_data)
