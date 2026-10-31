import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def test_10k_generation_answers_and_splits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records = [{"source": "reviewed", "source_id": str(i), "group": str(i),
                "messages": [{"role": "user", "content": f"Reviewed question {i}"},
                             {"role": "assistant", "content": "Answer"}]} for i in range(23)]
    (source / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for split in ("validation", "test"):
        (source / f"{split}.jsonl").write_text("")
    (source / "manifest.json").write_text("{}")
    output = tmp_path / "output"
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts" / "prepare_10k_pilot.py"),
               "--source", str(source), "--output", str(output)]
    subprocess.run(command, cwd=root, check=True, capture_output=True)
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    synthetic = [row for row in rows if row["source"] == "synthetic-en"]
    assert len(rows) == 10000 and len(synthetic) == 9977
    assert len({row["messages"][0]["content"].casefold() for row in synthetic}) == 9977
    assert len(Counter(row["family"] for row in synthetic)) == 10
    for row in synthetic:
        prompt, answer = [message["content"] for message in row["messages"]]
        family = row["family"]
        if family in {"addition", "subtraction", "multiplication", "parity"}:
            numbers = [int(x) for x in re.findall(r"\d+", prompt)]
            if family == "addition":
                expected = str(numbers[0] + numbers[1])
            elif family == "subtraction":
                expected = str(numbers[1] - numbers[0])
            elif family == "multiplication":
                expected = str(numbers[0] * numbers[1])
            else:
                expected = "Yes" if numbers[0] % 2 == 0 else "No"
            assert answer == expected
        elif family in {"uppercase", "lowercase", "reverse", "length"}:
            word = re.search(r"'([^']+)'", prompt).group(1)
            if family == "length":
                assert int(re.search(r"\d+", answer).group()) == len(word)
            else:
                expected = {"uppercase": word.upper(), "lowercase": word.lower(),
                            "reverse": word[::-1]}[family]
                assert answer == expected
        elif family == "sort":
            values = ast.literal_eval(re.search(r"\[[^]]+\]", prompt).group())
            assert ast.literal_eval(answer) == sorted(values)
        else:
            function = ast.parse(answer).body[0]
            assert isinstance(function, ast.FunctionDef)
            assert function.name == re.search(r"named (\w+)", prompt).group(1)
            expression = function.body[0].value
            assert isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)
            assert isinstance(expression.left, ast.Name) and expression.left.id == "x"
            assert ast.literal_eval(expression.right) == int(re.search(r"plus \((-?\d+)\)", prompt).group(1))
    for split in ("validation", "test"):
        assert (output / f"{split}.jsonl").read_bytes() == (source / f"{split}.jsonl").read_bytes()
    assert subprocess.run(command, cwd=root, check=False, capture_output=True).returncode != 0
