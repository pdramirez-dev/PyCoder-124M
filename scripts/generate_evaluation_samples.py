"""Regenerate saved evaluation prompts without executing the generated code."""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pycoder.config import ModelConfig
from pycoder.inference.generate import generate
from pycoder.model.gpt import GPT
from pycoder.training.trainer import select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("prompts", help="JSON file with a results list containing saved prompts")
    parser.add_argument("output", help="Destination for generated samples (must not exist)")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    cases = json.loads(Path(args.prompts).read_text(encoding="utf-8"))["results"]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device) if args.device else select_device()
    model = GPT(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    bos, eos = tokenizer.token_to_id("<|bos|>"), tokenizer.token_to_id("<|eos|>")
    if bos is None or eos is None:
        raise ValueError("Tokenizer must contain BOS and EOS tokens")
    samples = []
    for case in cases:
        mode = case["mode"]
        if mode not in {"greedy", "sample"}:
            raise ValueError(f"Unknown decoding mode: {mode}")
        torch.manual_seed(args.seed)
        ids = [bos, *tokenizer.encode(case["prompt"]).ids]
        tokens = generate(model, torch.tensor([ids], device=device), args.max_new_tokens,
                          0.8 if mode == "sample" else 1.0, 50 if mode == "sample" else 1,
                          eos, tokenizer.get_vocab_size())
        samples.append({key: case[key] for key in ("task", "format", "mode", "prompt")}
                       | {"output": tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)})
        print(f"{case['task']}: {case['format']} ({mode})", flush=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump({"seed": args.seed, "max_new_tokens": args.max_new_tokens,
                   "device": str(device), "results": samples}, file, indent=2, ensure_ascii=False)
        file.write("\n")


if __name__ == "__main__":
    main()
