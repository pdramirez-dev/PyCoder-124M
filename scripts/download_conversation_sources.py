"""Download pinned source shards for the small conversational data pilot."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

SOURCES = [
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k",
     "356bb069eee815daa6e23e9a282eeefe1490ad44",
     "data/train-00000-of-00001.parquet", "bigcode.parquet"),
    ("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b",
     "data/train-00000-of-00001-b42a775f407cee45.parquet", "oasst-train.parquet"),
    ("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b",
     "data/validation-00000-of-00001-134b8fd0c89408b6.parquet", "oasst-validation.parquet"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/conversation-pilot"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory")
    downloads = []
    for repo, revision, filename, target in SOURCES:
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        downloads.append((path, target, {"dataset": repo, "revision": revision,
                                        "file": filename, "sha256": digest}))
    args.output.mkdir(parents=True, exist_ok=False)
    for path, target, _ in downloads:
        shutil.copyfile(path, args.output / target)
    (args.output / "sources.json").write_text(
        json.dumps([item[2] | {"local_file": item[1]} for item in downloads], indent=2) + "\n",
        encoding="utf-8")
    print(f"Saved {len(downloads)} pinned source shards to {args.output}")


if __name__ == "__main__":
    main()
