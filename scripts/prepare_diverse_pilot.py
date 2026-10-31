"""Apply structural checks to the broader English pilot, retaining split boundaries."""

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

KNOWN_ISSUES = {
    "c73f7111-397a-4135-9e2e-48f937882209", "c8800cae-3cb4-4337-9254-ac096c83832b",
    "d133398f-6b80-4d5c-8644-989781f89f61", "c8c68f32-c838-4c45-a677-d090b471c7ac",
    "157560", "190969", "3687", "239953", "563206",
    "dcfad519-52b4-4450-a2b4-937ae0ba2f77", "cee189ea-1efd-48a0-aebf-ef860c9666ea",
    "6ac07256-f3ac-4a94-8e62-5c1d1a156cf9", "1dc77e26-e2e1-4bce-a8fd-06307e464892"
}


def rejection_reason(record):
    if record["source_id"] in KNOWN_ISSUES:
        return "known_review_issue"
    if record["source"] == "bigcode":
        blocks = re.findall(r"```python\s*\n(.*?)```", record["messages"][-1]["content"], re.DOTALL)
        if not blocks:
            return "missing_python_block"
        try:
            for block in blocks:
                ast.parse(block)
        except (SyntaxError, ValueError):
            return "invalid_python_block"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/conversation-pilot-en"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/conversation-diverse-en"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    raw = (args.source / "train.jsonl").read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    retained, rejected = [], Counter()
    for record in records:
        reason = rejection_reason(record)
        if reason:
            rejected[reason] += 1
        else:
            retained.append(record)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retained), encoding="utf-8")
    for split in ("validation", "test"):
        (args.output / f"{split}.jsonl").write_bytes((args.source / f"{split}.jsonl").read_bytes())
    manifest = {"source_train_sha256": hashlib.sha256(raw).hexdigest(),
                "counts": dict(Counter(r["source"] for r in retained)), "rejected": dict(rejected),
                "turn_counts": dict(Counter(len(r["messages"]) for r in retained)),
                "source_manifest": json.loads((args.source / "manifest.json").read_text(encoding="utf-8")),
                "note": "Structural filtering and known-issue exclusions only; not fully reviewed or decontaminated."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("counts", "rejected", "turn_counts")}, indent=2))


if __name__ == "__main__":
    main()
