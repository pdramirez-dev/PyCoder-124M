import numpy as np
import torch

from pycoder.config import TrainConfig
from pycoder.data.dataset import TokenStream
from pycoder.model.gpt import GPT
from pycoder.training.checkpoint import load_checkpoint, save_checkpoint
from pycoder.training.optimizer import make_optimizer, make_scheduler


def test_loss_decreases(tiny_config):
    torch.manual_seed(2)
    torch.set_num_threads(1)
    model = GPT(tiny_config)
    optim = torch.optim.AdamW(model.parameters(), lr=0.003)
    x = torch.tensor([[1, 2, 3, 4] * 4] * 2)
    y = torch.roll(x, -1, 1)
    losses = []
    for _ in range(20):
        optim.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_checkpoint_restores_state(tmp_path, tiny_config):
    path = tmp_path / "tokens.bin"
    np.arange(100, dtype=np.uint32).tofile(path)
    stream = TokenStream(str(path), tiny_config.context_length)
    config = TrainConfig()
    model = GPT(tiny_config)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    x, _ = stream.batch(2, torch.device("cpu"))
    _, loss = model(x % tiny_config.vocab_size, x % tiny_config.vocab_size)
    loss.backward()
    optimizer.step()
    scheduler.step()
    saved_weights = model.embedding.weight.detach().clone()
    save_checkpoint(tmp_path / "model.pt", model, optimizer, scheduler, scaler, 1, stream, config)
    expected_next = stream.batch(2, torch.device("cpu"))[0]
    with torch.no_grad():
        model.embedding.weight.zero_()
    assert load_checkpoint(tmp_path / "model.pt", model, optimizer, scheduler, scaler, stream, config) == 1
    assert torch.equal(model.embedding.weight, saved_weights)
    assert torch.equal(stream.batch(2, torch.device("cpu"))[0], expected_next)
    assert optimizer.state and scheduler.last_epoch == 1
