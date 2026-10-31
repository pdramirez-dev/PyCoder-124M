import json
import subprocess
import sys
from pathlib import Path


def test_prepare_100_preserves_splits_and_rejects_prompt_overlap(tmp_path):
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
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_100_pilot.py"
    command = [sys.executable, str(script), "--source", str(source), "--suite", str(suite),
               "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    result = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(result) == 100
    by_id = {row["source_id"]: row for row in result}
    assert by_id["synthetic-v1-addition-00"]["messages"][1]["content"] == "14"
    assert by_id["synthetic-v1-sort-00"]["messages"][1]["content"] == "[1, 3, 5]"
    assert (output / "validation.jsonl").read_bytes() == (source / "validation.jsonl").read_bytes()
    suite.write_text(json.dumps([{"prompt": by_id["synthetic-v1-addition-00"]["messages"][0]["content"]}]))
    command[-1] = str(tmp_path / "overlap")
    assert subprocess.run(command, check=False, capture_output=True).returncode != 0
