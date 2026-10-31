import pytest
import torch

from pycoder.inference.generate import find_stop, generate
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


def test_generation_stops_when_callback_requests_it(tiny_config):
    model = GPT(tiny_config)
    model.train()
    tokens = torch.tensor([[1, 2, 3]])
    seen = []

    def stop_at_two_new_tokens(generated):
        seen.append(generated.shape[1])
        return generated.shape[1] >= 5

    output = generate(model, tokens, max_new_tokens=10, stop_callback=stop_at_two_new_tokens)
    assert output.shape == (1, 5)
    assert seen == [4, 5]
    assert model.training


def test_stop_sequence_ignores_prompt_and_chooses_first_match():
    prompt = "def original():\n    "
    text = prompt + "return 1\n\nclass Extra:\n    pass\n\ndef other():"
    position = find_stop(text, len(prompt), ["\ndef ", "\nclass "])
    assert text[:position] == prompt + "return 1\n"
    assert find_stop(prompt, len(prompt), ["def "]) is None
    with pytest.raises(ValueError, match="empty"):
        find_stop(text, len(prompt), [""])
