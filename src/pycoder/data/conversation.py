"""Deterministic construction of a small, provenance-preserving chat pilot."""

import hashlib
from collections import Counter

ROLE_MARKERS = ("<|user|>", "<|assistant|>", "<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>")


def stable_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def usable_text(value):
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= 12000
            and not any(marker in value for marker in ROLE_MARKERS))


def build_code_examples(rows):
    """Group by seed hash so examples sharing a seed cannot cross splits."""
    examples, rejected = [], Counter()
    for row in rows:
        instruction, response, group = row.get("instruction"), row.get("response"), row.get("sha1")
        if not usable_text(instruction) or not usable_text(response) or not group:
            rejected["invalid_or_missing"] += 1
            continue
        examples.append({"source": "bigcode", "group": str(group), "source_id": str(row["id"]),
                         "messages": [{"role": "user", "content": instruction},
                                      {"role": "assistant", "content": response}]})
    return examples, rejected


def build_chat_examples(rows, language):
    """Reconstruct one reviewed path per tree in the requested language."""
    by_id = {row["message_id"]: row for row in rows}
    selected, rejected = {}, Counter()
    for row in rows:
        if row.get("role") != "assistant" or row.get("rank") not in (0, None):
            continue
        chain, seen, current = [], set(), row
        while current is not None:
            mid = current.get("message_id")
            if (not mid or mid in seen or current.get("lang") != language
                    or current.get("deleted") is not False
                    or current.get("review_result") is not True
                    or not usable_text(current.get("text"))):
                chain = []
                break
            seen.add(mid)
            chain.append(current)
            parent = current.get("parent_id")
            if parent and parent not in by_id:
                chain = []
                break
            current = by_id.get(parent) if parent else None
        chain.reverse()
        if (len(chain) < 2 or not row.get("message_tree_id")
                or any(r.get("message_tree_id") != row.get("message_tree_id")
                                  for r in chain)
                or any(r.get("role") != ("prompter" if i % 2 == 0 else "assistant")
                       for i, r in enumerate(chain))):
            rejected["invalid_path"] += 1
            continue
        group = row["message_tree_id"]
        item = {"source": f"oasst1-{language}", "group": group, "source_id": row["message_id"],
                "messages": [{"role": "user" if r["role"] == "prompter" else "assistant",
                              "content": r["text"]} for r in chain]}
        previous = selected.get(group)
        if previous is None or (-len(chain), item["source_id"]) < (
                -len(previous["messages"]), previous["source_id"]):
            selected[group] = item
    return list(selected.values()), rejected


def prepare_pilot(examples, max_groups_per_source=500):
    """Select groups and assign each example to train, validation or test."""
    if max_groups_per_source < 10:
        raise ValueError("At least 10 groups per source are required")
    groups = {}
    for item in examples:
        groups.setdefault((item["source"], item["group"]), []).append(item)
    sources = sorted({source for source, _ in groups})
    splits = {"train": [], "validation": [], "test": []}
    seen_content, duplicate = set(), 0
    for source in sources:
        keys = sorted((key for key in groups if key[0] == source),
                      key=lambda key: stable_hash(f"42:{key[0]}:{key[1]}"))
        keys = keys[:max_groups_per_source]
        for index, key in enumerate(keys):
            split = "test" if index % 10 == 0 else "validation" if index % 10 == 1 else "train"
            # A single example per group avoids overrepresenting a repeated seed.
            item = min(groups[key], key=lambda row: stable_hash(row["source_id"]))
            fingerprint = stable_hash(" ".join(item["messages"][0]["content"].split()).casefold())
            if fingerprint in seen_content:
                duplicate += 1
                continue
            seen_content.add(fingerprint)
            splits[split].append(item)
    return splits, {"duplicates_removed": duplicate,
                    "groups_selected_per_source": max_groups_per_source}
