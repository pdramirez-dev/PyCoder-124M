"""Build a small deterministic chat pilot without altering the pretrained data."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from pycoder.data.conversation import build_chat_examples, build_code_examples, prepare_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/conversation-pilot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "es"), default="en")
    parser.add_argument("--groups-per-source", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    sources = json.loads((args.source / "sources.json").read_text(encoding="utf-8"))
    for entry in sources:
        with (args.source / entry["local_file"]).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"Source hash mismatch: {entry['local_file']}")
    code, code_rejected = build_code_examples(pq.read_table(args.source / "bigcode.parquet").to_pylist())
    chat_rows = []
    for filename in ("oasst-train.parquet", "oasst-validation.parquet"):
        chat_rows.extend(pq.read_table(args.source / filename).to_pylist())
    chat, chat_rejected = build_chat_examples(chat_rows, args.language)
    splits, info = prepare_pilot(code + chat, args.groups_per_source)
    args.output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split, items in splits.items():
        with (args.output / f"{split}.jsonl").open("x", encoding="utf-8") as file:
            for item in items:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[split] = dict(Counter(item["source"] for item in items))
    chat_source = f"oasst1-{args.language}"
    manifest = {"language": args.language, "sources": sources,
                "eligible": {"bigcode": len(code), chat_source: len(chat)},
                "rejected": {"bigcode": dict(code_rejected), chat_source: dict(chat_rejected)},
                "selection": info, "counts": counts,
                "note": "Pilot requires quality, licensing and benchmark-overlap review."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"eligible": manifest["eligible"], "counts": counts,
                      "selection": info}, indent=2))


if __name__ == "__main__":
    main()
