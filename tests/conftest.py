import pytest

from pycoder.config import ModelConfig


@pytest.fixture
def tiny_config():
    return ModelConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=4,
                       d_model=32, ffn_dim=64)
