import runpy
from collections import Counter
from pathlib import Path

import pytest

sampler_class = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                   / "scripts" / "train_conversation_pilot.py"))["SourceSampler"]


def test_sampler_is_deterministic_and_respects_source_limits():
    train = [("reviewed", i) for i in range(3)] + [("synthetic", i) for i in range(100)]
    mixture = {"weights": {"reviewed": 2, "synthetic": 1}, "max_updates": {"reviewed": 6}}
    a, b = sampler_class(train, mixture), sampler_class(train, mixture)
    samples = [a.next() for _ in range(40)]
    assert samples == [b.next() for _ in range(40)]
    assert Counter(source for source, _ in samples) == {"reviewed": 6, "synthetic": 34}
    reviewed = [item for source, item in samples if source == "reviewed"]
    assert set(reviewed[:3]) == set(reviewed[3:]) == {0, 1, 2}


def test_sampler_rejects_missing_weights_and_exhaustion():
    train = [("a", 1), ("b", 2)]
    with pytest.raises(ValueError):
        sampler_class(train, {"weights": {"a": 1}})
    sampler = sampler_class(train, {"weights": {"a": 1, "b": 1},
                                    "max_updates": {"a": 0, "b": 0}})
    with pytest.raises(ValueError, match="exhausted"):
        sampler.next()
