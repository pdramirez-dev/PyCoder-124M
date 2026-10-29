from torch import nn
from torch.nn import functional as F

from pycoder.config import ModelConfig
from pycoder.model.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.RMSNorm(config.d_model)
        self.gate = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        h = self.ffn_norm(x)
        return x + self.dropout(self.down(F.silu(self.gate(h)) * self.up(h)))
