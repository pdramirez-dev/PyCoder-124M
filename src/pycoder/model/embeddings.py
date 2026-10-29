import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(-2), device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
