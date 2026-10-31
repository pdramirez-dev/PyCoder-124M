import argparse

import torch
from tokenizers import Tokenizer

from pycoder.config import ModelConfig
from pycoder.inference.generate import find_stop, generate
from pycoder.model.gpt import GPT
from pycoder.training.trainer import select_device

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--tokenizer", default="data/tokenizer.json")
parser.add_argument("--prompt", default="def ")
parser.add_argument("--max-new-tokens", type=int, default=100)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--stop", action="append", default=[], help="Stop at a generated text sequence")
parser.add_argument("--single-function", action="store_true",
                    help="Stop before another top-level function or class")
args = parser.parse_args()
if any(not stop for stop in args.stop):
    parser.error("--stop cannot be empty")
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
device = select_device()
model = GPT(ModelConfig(**state["model_config"])).to(device)
model.load_state_dict(state["model"])
tok = Tokenizer.from_file(args.tokenizer)
ids = [tok.token_to_id("<|bos|>"), *tok.encode(args.prompt).ids]
prompt_text = tok.decode(ids, skip_special_tokens=True)
stops = args.stop + (["\ndef ", "\nclass "] if args.single_function else [])


def stop_position(text):
    return find_stop(text, len(prompt_text), stops)


def should_stop(tokens):
    return stop_position(tok.decode(tokens[0].tolist(), skip_special_tokens=True)) is not None


result = generate(model, torch.tensor([ids], device=device), args.max_new_tokens,
                  args.temperature, args.top_k, tok.token_to_id("<|eos|>"), tok.get_vocab_size(),
                  should_stop if stops else None)
text = tok.decode(result[0].tolist(), skip_special_tokens=True)
position = stop_position(text)
print(text[:position] if position is not None else text)
