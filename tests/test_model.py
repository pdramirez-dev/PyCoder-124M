import pytest
import torch

from pycoder.inference.generate import generate
from pycoder.model.gpt import GPT


def test_forward_backward_and_generation(tiny_config):
    model = GPT(tiny_config)
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 8))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 8, tiny_config.vocab_size)
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert model.lm_head.weight is model.embedding.weight
    assert generate(model, tokens, 3).shape == (2, 11)


def test_mps(tiny_config):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = GPT(tiny_config).to("mps")
    x = torch.randint(0, tiny_config.vocab_size, (2, 8), device="mps")
    _, loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
