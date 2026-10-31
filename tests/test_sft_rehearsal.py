import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from pycoder.data.tokenizer import train_tokenizer
from pycoder.model.gpt import GPT


def test_sft_rehearsal_preserves_base_and_alternates_updates(tmp_path, tiny_config):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(["hello world"], str(tokenizer_path), 260)
    config = replace(tiny_config, vocab_size=tokenizer.get_vocab_size(), context_length=128)
    model = GPT(config)
    base = tmp_path / "base.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(config)}, base)
    original = base.read_bytes()
    row = {"source": "chat", "messages": [{"role": "user", "content": "Hello"},
                                            {"role": "assistant", "content": "Hello there"}]}
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    rehearsal = tmp_path / "code.bin"
    np.arange(64, dtype=np.uint32).tofile(rehearsal)
    output = tmp_path / "result.pt"
    final_output = tmp_path / "final.pt"
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_conversation_pilot.py"
    subprocess.run([sys.executable, str(script), str(base), "--data", str(tmp_path),
                    "--tokenizer", str(tokenizer_path), "--output", str(output),
                    "--final-output", str(final_output),
                    "--device", "cpu", "--steps", "2", "--rehearsal", str(rehearsal),
                    "--rehearsal-context", "16"], check=True, capture_output=True)
    result = torch.load(output, map_location="cpu", weights_only=False)["sft_pilot"]
    assert result["step_sources"] == ["chat", "code_rehearsal"]
    assert result["sft_updates"] == 1
    assert result["rehearsal_sha256"]
    assert torch.load(final_output, map_location="cpu", weights_only=False)["sft_pilot"]["saved_weights_step"] == 2
    assert base.read_bytes() == original
