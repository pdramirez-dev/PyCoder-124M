import math

import torch
from torch import nn
from torch.nn import functional as F

from pycoder.config import ModelConfig
from pycoder.model.embeddings import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, use_sdpa: bool = True):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_base)
        self.dropout = config.dropout
        self.use_sdpa = use_sdpa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = self.rope(q), self.rope(k)
        dropout = self.dropout if self.training else 0.0
        out = None
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            try:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)
            except NotImplementedError:
                pass  # Backend has no SDPA implementation; use the explicit causal path.
        if out is None:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            probs = F.dropout(scores.softmax(dim=-1), p=dropout, training=self.training)
            out = probs @ v
        return self.proj(out.transpose(1, 2).contiguous().view(batch, length, width))
