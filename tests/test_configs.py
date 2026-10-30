from pathlib import Path

from pycoder.config import FilterConfig, ModelConfig, TrainConfig, load_config
from pycoder.model.gpt import GPT

CONFIGS = Path(__file__).parent.parent / "configs"


def test_configs_load():
    for name in ("model_124m", "model_tiny"):
        load_config(CONFIGS / f"{name}.yaml", ModelConfig)
    for name in ("train", "train_smoke", "train_124m"):
        load_config(CONFIGS / f"{name}.yaml", TrainConfig)
    load_config(CONFIGS / "filters.yaml", FilterConfig)


def test_production_budget():
    model = load_config(CONFIGS / "model_124m.yaml", ModelConfig)
    train = load_config(CONFIGS / "train_124m.yaml", TrainConfig)
    tokens = train.batch_size * train.gradient_accumulation * model.context_length * train.max_steps
    assert 100e6 <= tokens < 101e6
    assert train.warmup_steps < train.max_steps
    assert GPT(model).num_parameters() == 123_587_328
