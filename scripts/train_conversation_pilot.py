"""Run a short, fresh supervised chat trial from a trusted base checkpoint."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pycoder.config import ModelConfig
from pycoder.data.conversation_sft import CHAT_FORMAT, collate_conversations, encode_conversation
from pycoder.data.dataset import TokenStream
from pycoder.model.gpt import GPT
from pycoder.training.trainer import select_device


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_examples(path, tokenizer, context_length, format_version=CHAT_FORMAT):
    items, skipped = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        encoded = encode_conversation(record["messages"], tokenizer, context_length, format_version)
        if encoded is None:
            skipped += 1
        else:
            items.append((record["source"], encoded))
    return items, skipped


class SourceSampler:
    """Sample deterministic shuffled source cycles with explicit update limits."""

    def __init__(self, train, mixture, seed=42):
        self.rng = random.Random(seed)
        self.groups = {name: [item for source, item in train if source == name]
                       for name in sorted({source for source, _ in train})}
        self.weights = mixture["weights"]
        self.limits = mixture.get("max_updates", {})
        if (set(self.weights) != set(self.groups) or set(self.limits) - set(self.groups)
                or any(not isinstance(w, int) or isinstance(w, bool) or w <= 0
                       for w in self.weights.values())
                or any(not isinstance(n, int) or isinstance(n, bool) or n < 0
                       for n in self.limits.values())):
            raise ValueError("Invalid source weights or limits")
        self.orders = {name: list(range(len(items))) for name, items in self.groups.items()}
        self.used = {name: 0 for name in self.groups}
        for order in self.orders.values():
            self.rng.shuffle(order)

    def next(self):
        available = [name for name in sorted(self.groups)
                     if name not in self.limits or self.used[name] < self.limits[name]]
        if not available:
            raise ValueError("All source update limits exhausted")
        name = self.rng.choices(available, weights=[self.weights[n] for n in available])[0]
        count = self.used[name]
        order = self.orders[name]
        if count and count % len(order) == 0:
            self.rng.shuffle(order)
        self.used[name] += 1
        return name, self.groups[name][order[count % len(order)]]


@torch.no_grad()
def validation_losses(model, records, pad_id, device, samples_per_source=10):
    model.eval()
    metrics = {}
    for source in sorted({source for source, _ in records}):
        subset = [encoded for name, encoded in records if name == source][:samples_per_source]
        losses = []
        for item in subset:
            x, y = collate_conversations([item], pad_id)
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        metrics[source] = sum(losses) / len(losses)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data", type=Path, default=Path("data/processed/conversation-pilot-en"))
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-output", type=Path,
                        help="Also save the final weights separately from the validation-selected weights")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate validation data this often and save the best-scoring weights")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rehearsal", type=Path, help="Original training token stream for code rehearsal")
    parser.add_argument("--rehearsal-context", type=int, default=256)
    parser.add_argument("--chat-format", choices=(CHAT_FORMAT, "legacy-v1"), default=CHAT_FORMAT)
    parser.add_argument("--source-mixture", type=Path,
                        help="JSON source weights and optional maximum SFT updates per source")
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_rate <= 0 or args.eval_every <= 0:
        parser.error("steps, eval-every, and learning rate must be positive")
    if args.output.exists():
        parser.error("output already exists")
    if args.final_output and (args.final_output.exists()
                              or args.final_output.resolve() == args.output.resolve()):
        parser.error("final-output must be a different, new path")
    if not 1 <= args.rehearsal_context <= 1024:
        parser.error("rehearsal-context must be between 1 and 1024")
    torch.manual_seed(42)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**state["model_config"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match checkpoint")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("Tokenizer has no padding token")
    train, skipped_train = read_examples(args.data / "train.jsonl", tokenizer, config.context_length, args.chat_format)
    validation, skipped_val = read_examples(args.data / "validation.jsonl", tokenizer, config.context_length, args.chat_format)
    if not train or not validation:
        raise ValueError("No usable training or validation examples")
    mixture = json.loads(args.source_mixture.read_text()) if args.source_mixture else None
    sampler = SourceSampler(train, mixture) if mixture else None
    device = torch.device(args.device) if args.device else select_device()
    model = GPT(config).to(device)
    model.load_state_dict(state["model"])
    rehearsal = None
    if args.rehearsal:
        if args.rehearsal_context > config.context_length:
            raise ValueError("Rehearsal context exceeds model context")
        rehearsal = TokenStream(str(args.rehearsal), args.rehearsal_context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    validation_samples_per_source = min(20, len(validation))
    before = validation_losses(model, validation, pad_id, device, validation_samples_per_source)
    best_model = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_validation = before
    best_score = sum(before.values()) / len(before)
    best_step = 0
    validation_history = [{"step": 0, "losses": before, "mean_source_loss": best_score}]
    order = list(range(len(train)))
    rng = random.Random(42)
    rng.shuffle(order)
    losses, step_sources = [], []
    chat_step = 0
    for step in range(args.steps):
        model.train()
        if rehearsal is not None and step % 2 == 1:
            x, y = rehearsal.batch(1, device)
            source = "code_rehearsal"
        else:
            if sampler:
                source, encoded = sampler.next()
            else:
                if chat_step and chat_step % len(order) == 0:
                    rng.shuffle(order)
                source, encoded = train[order[chat_step % len(order)]]
            x, y = collate_conversations([encoded], pad_id)
            chat_step += 1
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x.to(device), y.to(device))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss; no checkpoint was written")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        step_sources.append(source)
        print(f"step {step + 1}: {source} loss={loss.item():.4f}", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            current = validation_losses(model, validation, pad_id, device, validation_samples_per_source)
            score = sum(current.values()) / len(current)
            validation_history.append({"step": step + 1, "losses": current,
                                       "mean_source_loss": score})
            print(f"validation step {step + 1}: {json.dumps(current, sort_keys=True)}", flush=True)
            if score < best_score:
                best_score, best_validation, best_step = score, current, step + 1
                best_model = {name: tensor.detach().cpu().clone()
                              for name, tensor in model.state_dict().items()}
    final_model = ({name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                   if args.final_output else None)
    model.load_state_dict(best_model)
    report = {"kind": "fresh_sft_pilot_weights_only", "base_sha256": sha256(args.checkpoint),
              "chat_format": args.chat_format,
              "tokenizer_sha256": sha256(args.tokenizer),
              "train_sha256": sha256(args.data / "train.jsonl"),
              "validation_sha256": sha256(args.data / "validation.jsonl"),
              "steps": args.steps, "eval_every": args.eval_every,
              "source_mixture": mixture,
              "source_mixture_sha256": sha256(args.source_mixture) if args.source_mixture else None,
              "best_validation_step": best_step,
              "best_validation_mean_source_loss": best_score,
              "validation_history": validation_history,
              "seed": 42, "learning_rate": args.learning_rate,
              "batch_size": 1, "device": str(device), "train_loss": losses,
              "step_sources": step_sources, "sft_updates": chat_step,
              "rehearsal_sha256": sha256(args.rehearsal) if args.rehearsal else None,
              "rehearsal_context": args.rehearsal_context if args.rehearsal else None,
              "validation_before": before, "validation_at_selected_checkpoint": best_validation,
              "validation_samples_per_source": validation_samples_per_source,
              "usable_train": len(train), "usable_validation": len(validation),
              "skipped_long_train": skipped_train, "skipped_long_validation": skipped_val}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        torch.save({"model": model.state_dict(), "model_config": vars(config),
                    "sft_pilot": report}, file)
    if args.final_output:
        args.final_output.parent.mkdir(parents=True, exist_ok=True)
        with args.final_output.open("xb") as file:
            torch.save({"model": final_model, "model_config": vars(config),
                        "sft_pilot": {**report, "saved_weights_step": args.steps}}, file)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
