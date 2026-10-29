import hashlib
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def split_files(source: str, validation_fraction: float = 0.2, seed: int = 42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    files = sorted(Path(source).rglob("*.py"))
    unique = {}
    for path in files:
        content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode()).hexdigest()
        unique.setdefault(digest, (path, content))
    items = list(unique.values())
    if len(items) < 2:
        raise ValueError("At least two distinct Python files are required")
    random.Random(seed).shuffle(items)
    n_val = min(len(items) - 1, max(1, round(len(items) * validation_fraction)))
    return items[n_val:], items[:n_val]


def prepare(source: str, tokenizer_path: str, output: str, validation_fraction: float = 0.2,
            seed: int = 42):
    train_items, val_items = split_files(source, validation_fraction, seed)
    tok = Tokenizer.from_file(tokenizer_path)
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    if bos is None or eos is None:
        raise ValueError("Tokenizer must contain BOS and EOS")
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split, examples in (("train", train_items), ("val", val_items)):
        ids = []
        for _, content in examples:
            ids.extend([bos, *tok.encode(content).ids, eos])
        np.asarray(ids, dtype=np.uint32).tofile(dest / f"{split}.bin")
        manifest[split] = [str(path) for path, _ in examples]
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
