from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    n_layers: int
    n_heads: int
    d_model: int
    ffn_dim: int
    dropout: float = 0.0
    rope_base: float = 10000.0

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.n_layers, self.n_heads,
               self.d_model, self.ffn_dim) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("Head dimension must be even and divide d_model")
        if not 0 <= self.dropout < 1 or self.rope_base <= 0:
            raise ValueError("Invalid dropout or RoPE base")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 10
    max_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 20
    eval_batches: int = 10
    checkpoint_interval: int = 50
    precision: str = "float32"

    def __post_init__(self):
        if min(self.batch_size, self.gradient_accumulation, self.max_steps, self.eval_interval,
               self.eval_batches, self.checkpoint_interval) <= 0 or self.warmup_steps < 0:
            raise ValueError("Training counts must be positive (warmup may be zero)")
        if self.learning_rate <= 0 or not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("Invalid learning rates")
        if self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Unsupported precision")


def load_config(path: str | Path, cls):
    with open(path, encoding="utf-8") as file:
        return cls(**yaml.safe_load(file))
