"""Prepare a revised 100-example pilot with more varied synthetic answers."""

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_100_pilot import digest, normalized


def synthetic_examples():
    words = ["ox", "sun", "fern", "maple", "violet", "orchard", "elephant"]
    rows = []
    for i, word in enumerate(words):
        a, b = 13 + i * 3, 3 + i
        nums = [i + 8, i - 2, i + 2]
        length_prompt = (f"How many letters are in '{word}'? Reply with only the number."
                         if i % 2 == 0 else f"Count the letters in '{word}' in one sentence.")
        length_answer = str(len(word)) if i % 2 == 0 else f"There are {len(word)} letters."
        cases = [
            ("addition", f"Calculate {a} + {b}. Reply with only the number.", str(a + b)),
            ("subtraction", f"Subtract {b} from {a}. Reply with only the number.", str(a - b)),
            ("multiplication", f"Calculate {i + 2} times {i + 3}. Reply with only the number.",
             str((i + 2) * (i + 3))),
            ("uppercase", f"Convert '{word}' to uppercase. Output only the word.", word.upper()),
            ("lowercase", f"Convert '{word.upper()}' to lowercase. Output only the word.", word),
            ("length", length_prompt, length_answer),
            ("reverse", f"Write '{word}' backwards. Output only the word.", word[::-1]),
            ("sort", f"Put {nums} in ascending order. Output only the list.", str(sorted(nums))),
            ("parity", f"Is {a} even? Answer yes or no.", "Yes" if a % 2 == 0 else "No"),
            ("repeat", f"Write '{word}' twice separated by a space. Output only the result.",
             f"{word} {word}"),
            ("first_letter", f"What is the first letter of '{word}'? Output only that letter.", word[0]),
        ]
        for family, prompt, answer in cases:
            identifier = f"synthetic-v2-{family}-{i:02d}"
            rows.append({"source": "synthetic-en", "group": identifier, "source_id": identifier,
                         "messages": [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": answer}]})
    return rows


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
    heldout = []
    for split in ("validation", "test"):
        heldout.extend(json.loads(line) for line in (args.source / f"{split}.jsonl").read_text().splitlines())
    forbidden = {normalized(m["content"]) for row in train + heldout for m in row["messages"]
                 if m["role"] == "user"}
    forbidden.update(normalized(row["prompt"]) for row in json.loads(args.suite.read_text()))
    prompts = [normalized(row["messages"][0]["content"]) for row in synthetic]
    if len(synthetic) != 77 or len(set(prompts)) != 77 or set(prompts) & forbidden:
        raise ValueError("Synthetic prompts are duplicated or overlap an existing exact prompt")
    answers = Counter(row["messages"][1]["content"] for row in synthetic)
    if max(answers.values()) > 7:
        raise ValueError("A synthetic answer appears more than seven times")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "train.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train + synthetic),
                      encoding="utf-8")
    for split in ("validation", "test"):
        (args.output / f"{split}.jsonl").write_bytes((args.source / f"{split}.jsonl").read_bytes())
    manifest = {"generator": "synthetic-v2", "source_train_sha256": digest(args.source / "train.jsonl"),
                "suite_sha256": digest(args.suite), "train_sha256": digest(output),
                "counts": dict(Counter(row["source"] for row in train + synthetic)),
                "synthetic_families": 11, "synthetic_per_family": 7,
                "synthetic_answer_frequencies": dict(answers),
                "limitations": "Template-generated tasks; exact checks do not detect semantic overlap.",
                "source_manifest": json.loads((args.source / "manifest.json").read_text())}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "most_common_answers": answers.most_common(5)}))


if __name__ == "__main__":
    main()
