"""Run a small English completion-choice diagnostic on a trusted checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pycoder.config import ModelConfig
from pycoder.evaluation.choices import score_choice
from pycoder.model.gpt import GPT
from pycoder.training.trainer import select_device


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--suite", default="configs/english_choice_probe.json")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device) if args.device else select_device()
    model = GPT(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary differs from model")
    results = []
    for case in suite:
        scores = [score_choice(model, tokenizer, case["prompt"], option, device)
                  for option in case["options"]]
        predictions = {metric: min(range(len(scores)), key=lambda i: scores[i][metric])
                       for metric in ("nll_mean", "nll_sum")}
        results.append(case | {"scores": scores, "predictions": predictions})
    totals = {metric: sum(row["predictions"][metric] == row["correct"] for row in results)
              for metric in ("nll_mean", "nll_sum")}
    report = {"checkpoint_sha256": digest(args.checkpoint), "suite_sha256": digest(args.suite),
              "tokenizer_sha256": digest(args.tokenizer), "torch_version": torch.__version__,
              "device": str(device), "correct": totals, "total": len(results), "results": results}
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"correct": totals, "total": len(results)}))


if __name__ == "__main__":
    main()
