import torch
from torch import nn
from torch.nn import functional as F

from pycoder.config import ModelConfig
from pycoder.model.transformer import TransformerBlock


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2 or tokens.size(1) > self.config.context_length or tokens.size(1) == 0:
            raise ValueError("Expected nonempty [batch, sequence] within context length")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                            targets.reshape(-1))
        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
