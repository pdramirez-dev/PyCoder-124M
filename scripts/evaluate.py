import argparse

import torch

from pycoder.config import ModelConfig
from pycoder.data.dataset import TokenStream
from pycoder.model.gpt import GPT
from pycoder.training.trainer import evaluate, select_device

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--data", default="data/processed/val.bin")
parser.add_argument("--batches", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
config = ModelConfig(**state["model_config"])
device = select_device()
model = GPT(config).to(device)
model.load_state_dict(state["model"])
loss = evaluate(model, TokenStream(args.data, config.context_length), args.batch_size, args.batches, device)
print(f"validation loss: {loss:.4f}; perplexity: {torch.exp(torch.tensor(loss)).item():.2f}")
