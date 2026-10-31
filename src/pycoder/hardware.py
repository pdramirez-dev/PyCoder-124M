"""Detect hardware and suggest a conservative profile for the 124M model."""

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from pycoder.config import TrainConfig

GIB = 1024**3


@dataclass(frozen=True)
class Hardware:
    device: str
    name: str
    ram_gib: float | None
    vram_gib: float | None = None
    free_vram_gib: float | None = None
    bf16: bool = False


def system_ram_gib():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
    except (AttributeError, OSError, ValueError):
        try:
            result = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(result) / GIB
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None


def detect_hardware(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    ram = system_ram_gib()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available in this environment")
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        return Hardware(device, torch.cuda.get_device_name(index), ram,
                        total / GIB, free / GIB, torch.cuda.is_bf16_supported())
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available in this environment")
        return Hardware(device, "Apple GPU (unified memory)", ram)
    if device == "cpu":
        return Hardware(device, "CPU", ram)
    raise ValueError(f"Unknown device: {device}")


def suggest_profile(hardware):
    """Heuristic for model_124m.yaml at context 1024; never a memory guarantee."""
    batch = 1
    precision = "float32"
    notes = []
    if hardware.device == "cuda":
        available = min(hardware.vram_gib or 0, hardware.free_vram_gib or 0)
        # Reserve headroom; use free VRAM so other processes reduce the batch.
        budget = max(0, available - 1)
        for threshold, candidate in ((10, 2), (16, 4), (28, 8)):
            if budget >= threshold:
                batch = candidate
        precision = "bfloat16" if hardware.bf16 else "float16"
        if available < 4:
            notes.append("Low free VRAM: the 124M model may not fit even with batch size 1.")
    elif hardware.device == "cpu":
        notes.append("Training 124M on CPU may be very slow.")
    if hardware.ram_gib is None:
        notes.append("Could not determine system RAM.")
    elif hardware.ram_gib < (16 if hardware.device == "mps" else 8):
        notes.append("Limited RAM: monitor memory when loading checkpoints and preparing data.")
    notes.append("Estimate for 124M/context 1024; verify memory usage with a short trial.")
    config = TrainConfig(
        batch_size=batch, gradient_accumulation=64 // batch,
        learning_rate=0.0006, min_learning_rate=0.00006, warmup_steps=75,
        max_steps=1526, eval_interval=100, eval_batches=20, checkpoint_interval=50,
        precision=precision,
    )
    return config, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, help="Save a new YAML file without overwriting existing files")
    args = parser.parse_args()
    try:
        hardware = detect_hardware(args.device)
    except ValueError as error:
        parser.error(str(error))
    config, notes = suggest_profile(hardware)
    print(f"Device: {hardware.device} — {hardware.name}")
    if hardware.ram_gib is not None:
        print(f"Total RAM visible to the system: {hardware.ram_gib:.1f} GiB")
    if hardware.vram_gib is not None:
        print(f"VRAM: {hardware.vram_gib:.1f} GiB; free: {hardware.free_vram_gib:.1f} GiB")
    print("\nSuggested configuration:\n" + yaml.safe_dump(vars(config), sort_keys=False))
    for note in notes:
        print(f"- {note}")
    print("- To resume, keep the original checkpoint configuration.")
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as file:
                yaml.safe_dump(vars(config), file, sort_keys=False)
        except OSError as error:
            parser.error(str(error))
        print(f"\nProfile saved: {args.output}")


if __name__ == "__main__":
    main()
