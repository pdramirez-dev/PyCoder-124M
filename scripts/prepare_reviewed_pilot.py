"""Select explicitly reviewed training examples without changing held-out splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/conversation-pilot-en"))
    parser.add_argument("--review", type=Path, default=Path("configs/conversation_review_en.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/conversation-reviewed-en"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    review = json.loads(args.review.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (args.source / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    chosen = [record for record in records if record["source_id"] in review["approved"]]
    if (len(chosen) != len(review["approved"])
            or {record["source_id"] for record in chosen} != set(review["approved"])):
        raise ValueError("Reviewed IDs do not match source training data")
    for split in ("validation", "test"):
        heldout = [json.loads(line) for line in (args.source / f"{split}.jsonl")
                   .read_text(encoding="utf-8").splitlines()]
        if {(r["source"], r["group"]) for r in chosen} & {
            (r["source"], r["group"]) for r in heldout
        }:
            raise ValueError(f"Reviewed training groups overlap with {split}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    for split in ("validation", "test"):
        (args.output / f"{split}.jsonl").write_bytes((args.source / f"{split}.jsonl").read_bytes())
    manifest = {"review": review, "train_counts": dict(Counter(r["source"] for r in chosen)),
                "source_train_sha256": hashlib.sha256((args.source / "train.jsonl").read_bytes()).hexdigest(),
                "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
                "heldout_note": "Original held-out splits unchanged and not reviewed as training examples",
                "source_manifest": json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["train_counts"]))


if __name__ == "__main__":
    main()
