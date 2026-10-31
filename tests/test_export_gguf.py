import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pycoder.data.tokenizer import train_tokenizer
from pycoder.model.gpt import GPT


def test_export_preserves_qkv_and_embeddings(tmp_path, tiny_config):
    gguf = pytest.importorskip("gguf")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(["def add(a, b):\n    return a + b\n"], str(tokenizer_path), 300)
    config = replace(tiny_config, vocab_size=tokenizer.get_vocab_size())
    model = GPT(config)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": asdict(config), "model": model.state_dict()}, checkpoint)
    output = tmp_path / "model.gguf"
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_gguf.py"
    subprocess.run([sys.executable, str(script), str(checkpoint), str(output),
                    "--tokenizer", str(tokenizer_path)], check=True)
    reader = gguf.GGUFReader(output)
    tensors = {tensor.name: tensor.data for tensor in reader.tensors}
    np.testing.assert_array_equal(tensors["token_embd.weight"], model.embedding.weight.detach())
    for name, expected in zip(("attn_q", "attn_k", "attn_v"),
                              model.blocks[0].attention.qkv.weight.detach().chunk(3), strict=True):
        np.testing.assert_array_equal(tensors[f"blk.0.{name}.weight"], expected)
    assert "output.weight" not in tensors
    assert reader.fields["llama.attention.layer_norm_rms_epsilon"].contents() == pytest.approx(
        np.finfo(np.float32).eps
    )
