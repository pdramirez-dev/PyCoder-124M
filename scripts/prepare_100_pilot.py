"""Prepare 100 English examples: reviewed conversations plus simple synthetic tasks."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_examples():
    """Create seven transparent task families, with eleven values per family."""
    words = ["maple", "river", "stone", "cloud", "brush", "flame", "ocean", "plant",
             "tiger", "pearl", "grass"]
    rows = []
    for i, word in enumerate(words):
        a, b = i + 12, (i * 3) % 10 + 2
        numbers = [i + 5, i + 1, i + 3]
        cases = [
            ("addition", f"What is {a} plus {b}? Reply with only the number.", str(a + b)),
            ("subtraction", f"What is {a} minus {b}? Reply with only the number.", str(a - b)),
            ("uppercase", f"Write '{word}' in uppercase. Reply with only the result.", word.upper()),
            ("lowercase", f"Write '{word.upper()}' in lowercase. Reply with only the result.", word),
            ("length", f"How many letters are in '{word}'? Reply with only the number.", str(len(word))),
            ("reverse", f"Reverse the letters in '{word}'. Reply with only the result.", word[::-1]),
            ("sort", f"Sort these numbers in ascending order: {numbers}. Reply with only the list.",
             str(sorted(numbers))),
        ]
        for family, prompt, answer in cases:
            identifier = f"synthetic-v1-{family}-{i:02d}"
            rows.append({"source": "synthetic-en", "group": identifier, "source_id": identifier,
                         "messages": [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": answer}]})
    return rows


def normalized(value):
    return " ".join(value.split()).casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/conversation-reviewed-23-en"))
    parser.add_argument("--suite", type=Path, default=Path("configs/chat_eval_en.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    train = [json.loads(line) for line in (args.source / "train.jsonl").read_text().splitlines()]
    if len(train) != 23 or len({r["source_id"] for r in train}) != 23:
        raise ValueError("Expected 23 unique reviewed records")
    synthetic = synthetic_examples()
    holdout = []
    for split in ("validation", "test"):
        holdout.extend(json.loads(line) for line in (args.source / f"{split}.jsonl").read_text().splitlines())
    forbidden = {normalized(m["content"]) for r in train + holdout for m in r["messages"]
                 if m["role"] == "user"}
    forbidden.update(normalized(r["prompt"]) for r in json.loads(args.suite.read_text()))
    prompts = [normalized(r["messages"][0]["content"]) for r in synthetic]
    if len(synthetic) != 77 or len(set(prompts)) != 77 or set(prompts) & forbidden:
        raise ValueError("Synthetic prompts are duplicated or overlap an existing exact prompt")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "train.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train + synthetic),
                      encoding="utf-8")
    for split in ("validation", "test"):
        (args.output / f"{split}.jsonl").write_bytes((args.source / f"{split}.jsonl").read_bytes())
    manifest = {"source_train_sha256": digest(args.source / "train.jsonl"),
                "suite_sha256": digest(args.suite), "train_sha256": digest(output),
                "counts": dict(Counter(r["source"] for r in train + synthetic)),
                "synthetic_families": 7, "synthetic_per_family": 11,
                "limitations": "Synthetic examples are generated from simple templates; exact prompt "
                               "checks do not establish absence of paraphrase or pretraining overlap.",
                "source_manifest": json.loads((args.source / "manifest.json").read_text())}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
