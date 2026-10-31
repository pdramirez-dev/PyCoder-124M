import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from pycoder.data.tokenizer import train_tokenizer
from pycoder.model.gpt import GPT


@pytest.mark.parametrize("format_version", ["legacy-v1", "turn-eos-v2"])
def test_chat_evaluation_saves_responses_and_refuses_overwrite(tmp_path, tiny_config, format_version):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(["Hello, how can I help you?"], str(tokenizer_path), 300)
    config = replace(tiny_config, vocab_size=tokenizer.get_vocab_size(), context_length=128)
    model = GPT(config)
    checkpoint = tmp_path / "model.pt"
    state = {"model_config": asdict(config), "model": model.state_dict()}
    if format_version != "legacy-v1":
        state["sft_pilot"] = {"chat_format": format_version}
    torch.save(state, checkpoint)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps([{"id": "hello", "prompt": "Hello", "review": "manual"}]))
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_chat_samples.py"
    command = [sys.executable, str(script), str(checkpoint), str(output), "--suite", str(suite),
               "--tokenizer", str(tokenizer_path), "--device", "cpu", "--max-new-tokens", "2"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["chat_format"] == format_version
    assert result["results"][0]["id"] == "hello"
    assert result["results"][0]["finish_reason"] in {"eos", "token_limit", "role_stop"}
    original = output.read_bytes()
    assert subprocess.run(command, check=False, capture_output=True).returncode != 0
    assert output.read_bytes() == original
