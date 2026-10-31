import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def test_v2_varies_lengths_and_parity_and_preserves_heldout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records = [{"source": "reviewed", "source_id": str(i), "group": str(i),
                "messages": [{"role": "user", "content": f"Question {i}"},
                             {"role": "assistant", "content": "Answer"}]} for i in range(23)]
    (source / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for split in ("validation", "test"):
        (source / f"{split}.jsonl").write_text("")
    (source / "manifest.json").write_text("{}")
    suite = tmp_path / "suite.json"
    suite.write_text("[]")
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_100_pilot_v2.py"
    command = [sys.executable, str(script), "--source", str(source), "--suite", str(suite),
               "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    synthetic = [row for row in rows if row["source"] == "synthetic-en"]
    assert len(rows) == 100 and len(synthetic) == 77
    lengths = [row for row in synthetic if "-length-" in row["source_id"]]
    for row in lengths:
        prompt, answer = [message["content"] for message in row["messages"]]
        word = re.search(r"'([^']+)'", prompt).group(1)
        assert int(re.search(r"\d+", answer).group()) == len(word)
    assert len({len(re.search(r"'([^']+)'", row["messages"][0]["content"]).group(1))
                for row in lengths}) == 7
    parity = [row for row in synthetic if "-parity-" in row["source_id"]]
    for row in parity:
        prompt, answer = [message["content"] for message in row["messages"]]
        assert answer == ("Yes" if int(re.search(r"\d+", prompt).group()) % 2 == 0 else "No")
    assert {row["messages"][1]["content"] for row in parity} == {"Yes", "No"}
    assert max(Counter(row["messages"][1]["content"] for row in synthetic).values()) <= 7
    for split in ("validation", "test"):
        assert (output / f"{split}.jsonl").read_bytes() == (source / f"{split}.jsonl").read_bytes()
    assert subprocess.run(command, check=False, capture_output=True).returncode != 0
