import contextlib
import random
import signal
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pycoder.data.dataset import TokenStream
from pycoder.model.gpt import GPT
from pycoder.training.checkpoint import load_checkpoint, save_checkpoint
from pycoder.training.optimizer import make_optimizer, make_scheduler


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device, precision):
    if precision == "float32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed precision is supported here only on CUDA; use float32")
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bfloat16")
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(model, stream, batch_size, batches, device, precision="float32"):
    was_training = model.training
    position = stream.position
    model.eval()
    losses = []
    try:
        for _ in range(batches):
            x, y = stream.batch(batch_size, device)
            with autocast_context(device, precision):
                _, loss = model(x, y)
            losses.append(loss.item())
    finally:
        stream.position = position
        model.train(was_training)
    return sum(losses) / len(losses)


def train(model_config, config, data_dir, checkpoint_dir, log_dir, resume=None, device=None):
    device = torch.device(device) if device else select_device()
    autocast_context(device, config.precision)  # validate before allocating model
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = GPT(model_config).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.precision == "float16")
    train_stream = TokenStream(str(Path(data_dir) / "train.bin"), model_config.context_length)
    val_stream = TokenStream(str(Path(data_dir) / "val.bin"), model_config.context_length)
    step = load_checkpoint(resume, model, optimizer, scheduler, scaler, train_stream, config) if resume else 0
    checkpoint = Path(checkpoint_dir) / "latest.pt"
    writer = SummaryWriter(log_dir)
    stop = False
    previous_handler = signal.getsignal(signal.SIGINT)
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("\nInterrupción solicitada: guardando al terminar el paso actual.")
    signal.signal(signal.SIGINT, request_stop)
    print(f"Device: {device}; parameters: {model.num_parameters():,}; start step: {step}")
    try:
        for _ in tqdm(range(step, config.max_steps), desc="Training"):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for _ in range(config.gradient_accumulation):
                x, y = train_stream.batch(config.batch_size, device)
                with autocast_context(device, config.precision):
                    _, loss = model(x, y)
                    scaled_loss = loss / config.gradient_accumulation
                scaler.scale(scaled_loss).backward()
                loss_total += loss.item() / config.gradient_accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            writer.add_scalar("loss/train", loss_total, step)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], step)
            if device.type == "mps":
                writer.add_scalar("memory/mps_allocated_bytes", torch.mps.current_allocated_memory(), step)
            elif device.type == "cuda":
                writer.add_scalar("memory/cuda_allocated_bytes", torch.cuda.memory_allocated(), step)
            if step % config.eval_interval == 0 or step == config.max_steps:
                val = evaluate(model, val_stream, config.batch_size, config.eval_batches, device, config.precision)
                writer.add_scalar("loss/validation", val, step)
                print(f"step {step}: train={loss_total:.4f} val={val:.4f}")
            if step % config.checkpoint_interval == 0 or stop or step == config.max_steps:
                save_checkpoint(checkpoint, model, optimizer, scheduler, scaler, step, train_stream, config)
            if stop:
                break
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        writer.close()
    return model, step
