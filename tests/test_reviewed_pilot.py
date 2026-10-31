import json
import subprocess
import sys
from pathlib import Path


def test_reviewed_selection_keeps_heldout_files_exactly(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [{"source_id": "approved", "source": "test", "group": "train", "messages": []},
            {"source_id": "other", "source": "test", "group": "other", "messages": []}]
    (source / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for split in ("validation", "test"):
        (source / f"{split}.jsonl").write_text(json.dumps({"source": "test", "group": split}) + "\n")
    (source / "manifest.json").write_text("{}")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"approved": {"approved": "reviewed"}}))
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_reviewed_pilot.py"
    command = [sys.executable, str(script), "--source", str(source), "--review", str(review),
               "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    chosen = [json.loads(row) for row in (output / "train.jsonl").read_text().splitlines()]
    assert [row["source_id"] for row in chosen] == ["approved"]
    for split in ("validation", "test"):
        assert (output / f"{split}.jsonl").read_bytes() == (source / f"{split}.jsonl").read_bytes()
    assert subprocess.run(command, check=False, capture_output=True).returncode != 0
    (source / "validation.jsonl").write_text(json.dumps({"source": "test", "group": "train"}) + "\n")
    command[-1] = str(tmp_path / "overlap")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "overlap" in result.stderr
