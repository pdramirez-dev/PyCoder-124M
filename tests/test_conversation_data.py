from pycoder.data.conversation import build_chat_examples, build_code_examples, prepare_pilot


def message(mid, parent, role, tree="tree", **overrides):
    return {"message_id": mid, "parent_id": parent, "message_tree_id": tree,
            "role": role, "lang": "es", "deleted": False, "review_result": True,
            "rank": 0, "text": f"message {mid}"} | overrides


def test_spanish_paths_require_reviewed_complete_tree():
    rows = [message("u", None, "prompter"), message("a", "u", "assistant"),
            message("u2", "a", "prompter"), message("a2", "u2", "assistant"),
            message("bad", "missing", "assistant", tree="other"),
            message("bad2", "u", "assistant", review_result=False)]
    examples, rejected = build_chat_examples(rows, "es")
    assert len(examples) == 1
    assert examples[0]["source_id"] == "a2"
    assert [m["role"] for m in examples[0]["messages"]] == ["user", "assistant"] * 2
    assert rejected["invalid_path"] == 2
    assert build_chat_examples(rows, "en")[0] == []


def test_english_paths_require_same_language():
    rows = [message("u", None, "prompter", lang="en"),
            message("a", "u", "assistant", lang="en"),
            message("a2", "u", "assistant", lang="es")]
    examples, _ = build_chat_examples(rows, "en")
    assert len(examples) == 1
    assert examples[0]["source"] == "oasst1-en"


def test_split_deterministic_and_prevents_duplicate_prompt_leakage():
    rows = [{"id": i, "sha1": str(i), "instruction": f"question {i}", "response": "answer"}
            for i in range(20)]
    rows[11]["instruction"] = rows[0]["instruction"]
    rows.append({"id": 21, "sha1": "bad", "instruction": "<|user|>", "response": "bad"})
    items, rejected = build_code_examples(rows)
    assert rejected["invalid_or_missing"] == 1
    splits, info = prepare_pilot(items, max_groups_per_source=20)
    assert (splits, info) == prepare_pilot(list(reversed(items)), max_groups_per_source=20)
    all_items = [item for split in splits.values() for item in split]
    prompts = [item["messages"][0]["content"] for item in all_items]
    assert len(prompts) == len(set(prompts))
    assert info["duplicates_removed"] == 1
