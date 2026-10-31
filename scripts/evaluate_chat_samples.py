"""Generate fixed English chat samples for manual review; never execute them."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pycoder.config import ModelConfig
from pycoder.data.conversation_sft import CHAT_FORMAT, format_turns
from pycoder.inference.generate import find_stop, generate
from pycoder.model.gpt import GPT
from pycoder.training.trainer import select_device


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def chat_prompt(prompt, format_version="legacy-v1"):
    if not isinstance(prompt, str) or not prompt.strip() or "<|" in prompt:
        raise ValueError("Invalid prompt")
    if format_version == CHAT_FORMAT:
        return format_turns([{"role": "user", "content": prompt}], add_generation_prompt=True)[0]
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--suite", default="configs/chat_eval_en.json")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--chat-format", choices=("auto", CHAT_FORMAT, "legacy-v1"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    cases = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    format_version = (state.get("sft_pilot", {}).get("chat_format", "legacy-v1")
                      if args.chat_format == "auto" else args.chat_format)
    config = ModelConfig(**state["model_config"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary differs from model")
    bos, eos = tokenizer.token_to_id("<|bos|>"), tokenizer.token_to_id("<|eos|>")
    if bos is None or eos is None:
        raise ValueError("Tokenizer lacks BOS or EOS")
    device = torch.device(args.device) if args.device else select_device()
    model = GPT(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    results = []
    stops = ["<|user|>", "<|assistant|>"]
    for case in cases:
        prompt = chat_prompt(case["prompt"], format_version)
        prompt_ids = [bos, *tokenizer.encode(prompt, add_special_tokens=False).ids]
        if len(prompt_ids) + args.max_new_tokens > config.context_length:
            raise ValueError(f"Prompt and output exceed context: {case['id']}")
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        prompt_length = len(prompt_text)

        def should_stop(tokens, prompt_length=prompt_length):
            decoded = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
            return find_stop(decoded, prompt_length, stops) is not None

        torch.manual_seed(42)
        output = generate(model, torch.tensor([prompt_ids], device=device), args.max_new_tokens,
                          temperature=1.0, top_k=1, eos_id=eos,
                          vocab_limit=tokenizer.get_vocab_size(), stop_callback=should_stop)
        generated = output[0].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        position = find_stop(text, len(prompt_text), stops)
        reason = "role_stop" if position is not None else (
            "eos" if generated[-1] == eos else "token_limit")
        reply = text[len(prompt_text):position].strip() if position is not None else text[len(prompt_text):].strip()
        results.append(case | {"reply": reply, "finish_reason": reason,
                               "generated_tokens": len(generated) - len(prompt_ids)})
        print(f"{case['id']}: {reason}", flush=True)
    report = {"checkpoint_sha256": file_hash(args.checkpoint),
              "chat_format": format_version,
              "tokenizer_sha256": file_hash(args.tokenizer), "suite_sha256": file_hash(args.suite),
              "torch_version": torch.__version__, "device": str(device), "seed": 42,
              "top_k": 1, "max_new_tokens": args.max_new_tokens, "results": results}
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")


if __name__ == "__main__":
    main()
