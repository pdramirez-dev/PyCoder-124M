import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, stream, train_config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step,
             "position": stream.position, "data_hash": stream.fingerprint, "model_config": vars(model.config),
             "train_config": vars(train_config), "python_rng": random.getstate(),
             "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, stream, train_config):
    # Checkpoints include Python RNG state: load only trusted files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["model_config"] != vars(model.config) or state["train_config"] != vars(train_config):
        raise ValueError("Checkpoint configuration differs from current configuration")
    if state["data_hash"] != stream.fingerprint:
        raise ValueError("Training dataset differs from checkpoint")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    stream.position = state["position"]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if state["mps_rng"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps_rng"])
    return state["step"]
