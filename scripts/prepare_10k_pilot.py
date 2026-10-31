"""Generate a reproducible 10,000-row pilot with labeled synthetic examples."""

import argparse
import json
import random
import string
from collections import Counter
from pathlib import Path

from prepare_100_pilot import digest, normalized

FAMILIES = ("addition", "subtraction", "multiplication", "uppercase", "lowercase",
            "length", "reverse", "sort", "parity", "code_add_constant")


def make_case(family, rng):
    a, b = rng.randint(10, 9999), rng.randint(1, 999)
    word = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10)))
    if family == "addition":
        return f"Calculate {a} + {b}. Reply with only the number.", str(a + b)
    if family == "subtraction":
        return f"Subtract {b} from {a}. Reply with only the number.", str(a - b)
    if family == "multiplication":
        x, y = rng.randint(2, 200), rng.randint(2, 200)
        return f"Calculate {x} times {y}. Reply with only the number.", str(x * y)
    if family == "uppercase":
        return f"Convert '{word}' to uppercase. Output only the string.", word.upper()
    if family == "lowercase":
        return f"Convert '{word.upper()}' to lowercase. Output only the string.", word
    if family == "length":
        if rng.randrange(2):
            return f"Count the letters in '{word}' in one sentence.", f"There are {len(word)} letters."
        return f"How many letters are in '{word}'? Reply with only the number.", str(len(word))
    if family == "reverse":
        return f"Write '{word}' backwards. Output only the string.", word[::-1]
    if family == "sort":
        values = rng.sample(range(-1000, 1001), 3)
        return f"Put {values} in ascending order. Output only the list.", str(sorted(values))
    if family == "parity":
        return f"Is {a} even? Answer yes or no.", "Yes" if a % 2 == 0 else "No"
    if family == "code_add_constant":
        offset = rng.randint(-9999, 9999)
        name = f"offset_{'m' if offset < 0 else 'p'}{abs(offset)}"
        prompt = (f"Write a Python function named {name} that takes x and returns x plus "
                  f"({offset}). Output only the function.")
        return prompt, f"def {name}(x):\n    return x + ({offset})"
    raise ValueError(f"Unknown family: {family}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/conversation-reviewed-23-en"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    train = [json.loads(line) for line in (args.source / "train.jsonl").read_text().splitlines()]
    if len(train) != 23 or len({r["source_id"] for r in train}) != 23:
        raise ValueError("Expected 23 unique reviewed training records")
    existing = list(train)
    for split in ("validation", "test"):
        existing.extend(json.loads(line) for line in (args.source / f"{split}.jsonl").read_text().splitlines())
    seen = {normalized(m["content"]) for row in existing for m in row["messages"] if m["role"] == "user"}
    suites = [Path("configs/chat_eval_en.json"), Path("configs/synthetic_generalization_eval_en.json")]
    for path in suites:
        seen.update(normalized(case["prompt"]) for case in json.loads(path.read_text()))
    rng = random.Random(args.seed)
    rows, family_counts, rejected = [], Counter(), 0
    for index in range(9977):
        family = FAMILIES[index % len(FAMILIES)]
        for _ in range(10000):
            prompt, answer = make_case(family, rng)
            if normalized(prompt) not in seen:
                break
            rejected += 1
        else:
            raise RuntimeError(f"Could not create a unique prompt for {family}")
        seen.add(normalized(prompt))
        identifier = f"synthetic-10k-v1-{family}-{index:05d}"
        rows.append({"source": "synthetic-en", "group": identifier, "source_id": identifier,
                     "family": family, "messages": [{"role": "user", "content": prompt},
                                                    {"role": "assistant", "content": answer}]})
        family_counts[family] += 1
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "train.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train + rows),
                      encoding="utf-8")
    for split in ("validation", "test"):
        (args.output / f"{split}.jsonl").write_bytes((args.source / f"{split}.jsonl").read_bytes())
    answer_counts = Counter(row["messages"][1]["content"] for row in rows)
    manifest = {"generator": "synthetic-10k-v1", "generator_sha256": digest(Path(__file__)),
                "seed": args.seed, "source_train_sha256": digest(args.source / "train.jsonl"),
                "suite_sha256": {str(path): digest(path) for path in suites},
                "train_sha256": digest(output), "counts": dict(Counter(r["source"] for r in train + rows)),
                "synthetic_families": dict(family_counts), "duplicate_prompts_rejected": rejected,
                "most_common_synthetic_answers": answer_counts.most_common(10),
                "limitations": "Synthetic tasks use templates and invented strings; exact prompt checks "
                               "do not detect semantic or pretraining overlap.",
                "source_manifest": json.loads((args.source / "manifest.json").read_text())}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "families": dict(family_counts),
                      "most_common_answers": answer_counts.most_common(5)}, indent=2))


if __name__ == "__main__":
    main()
