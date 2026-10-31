import math
from types import SimpleNamespace

import pytest
import torch

from pycoder.evaluation.choices import score_choice


class CharacterTokenizer:
    def token_to_id(self, text):
        return 0

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[1] * len(text), offsets=[(i, i + 1) for i in range(len(text))])


class UniformModel:
    config = SimpleNamespace(context_length=64)

    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros((*x.shape, 4), device=x.device), None


def test_choice_scores_only_continuation_tokens():
    score = score_choice(UniformModel(), CharacterTokenizer(), "prompt", "answer", "cpu")
    assert score["tokens"] == len("answer")
    assert score["nll_mean"] == pytest.approx(math.log(4))
    assert score["nll_sum"] == pytest.approx(len("answer") * math.log(4))
    assert score_choice(UniformModel(), CharacterTokenizer(), "much longer prompt", "answer", "cpu") == score


def test_choice_rejects_empty_or_overlong_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        score_choice(UniformModel(), CharacterTokenizer(), "prompt", "", "cpu")
    with pytest.raises(ValueError, match="context"):
        score_choice(UniformModel(), CharacterTokenizer(), "x" * 64, "too long", "cpu")
