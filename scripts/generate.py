import argparse

import torch
from tokenizers import Tokenizer

from pycoder.config import ModelConfig
from pycoder.inference.generate import generate
from pycoder.model.gpt import GPT
from pycoder.training.trainer import select_device

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--tokenizer", default="data/tokenizer.json")
parser.add_argument("--prompt", default="def ")
parser.add_argument("--max-new-tokens", type=int, default=100)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
args = parser.parse_args()
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
device = select_device()
model = GPT(ModelConfig(**state["model_config"])).to(device)
model.load_state_dict(state["model"])
tok = Tokenizer.from_file(args.tokenizer)
ids = [tok.token_to_id("<|bos|>"), *tok.encode(args.prompt).ids]
result = generate(model, torch.tensor([ids], device=device), args.max_new_tokens,
                  args.temperature, args.top_k, tok.token_to_id("<|eos|>"), tok.get_vocab_size())
print(tok.decode(result[0].tolist(), skip_special_tokens=True))
