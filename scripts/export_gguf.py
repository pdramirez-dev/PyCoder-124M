"""Export a trusted PyCoder checkpoint to experimental float32 Llama GGUF."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    try:
        import gguf
    except ImportError:
        parser.error("install gguf or run with uv run --with gguf")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, weights = state["model_config"], state["model"]
    if not torch.equal(weights["embedding.weight"], weights["lm_head.weight"]):
        raise ValueError("Expected tied embedding and output weights")
    tokenizer = json.loads(Path(args.tokenizer).read_text(encoding="utf-8"))
    pre, model = tokenizer["pre_tokenizer"], tokenizer["model"]
    if (model["type"] != "BPE" or pre.get("type") != "ByteLevel"
            or pre.get("add_prefix_space") or not pre.get("use_regex", True)):
        raise ValueError("Expected PyCoder byte-level BPE tokenizer")
    vocab = model["vocab"]
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        if not isinstance(index, int) or not 0 <= index < len(tokens) or tokens[index] is not None:
            raise ValueError("Tokenizer IDs must be contiguous and unique")
        tokens[index] = token
    if len(tokens) != config["vocab_size"]:
        raise ValueError("Tokenizer vocabulary must match model vocabulary")
    special = {token["content"]: token["id"] for token in tokenizer["added_tokens"]
               if token.get("special")}
    if special != {"<|pad|>": 0, "<|unk|>": 1, "<|bos|>": 2, "<|eos|>": 3}:
        raise ValueError("Unexpected special tokens")

    writer = gguf.GGUFWriter(args.output, "llama", use_temp_file=True)
    writer.add_name("PyCoder experimental F32")
    writer.add_context_length(config["context_length"])
    writer.add_embedding_length(config["d_model"])
    writer.add_block_count(config["n_layers"])
    writer.add_feed_forward_length(config["ffn_dim"])
    writer.add_head_count(config["n_heads"])
    writer.add_head_count_kv(config["n_heads"])
    writer.add_rope_dimension_count(config["d_model"] // config["n_heads"])
    writer.add_rope_freq_base(config["rope_base"])
    writer.add_layer_norm_rms_eps(float(np.finfo(np.float32).eps))
    writer.add_file_type(0)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_merges([" ".join(pair) if isinstance(pair, list) else pair
                             for pair in model["merges"]])
    writer.add_token_types([gguf.TokenType.CONTROL if i in (0, 2, 3) else
                            gguf.TokenType.UNKNOWN if i == 1 else gguf.TokenType.NORMAL
                            for i in range(len(tokens))])
    writer.add_pad_token_id(0)
    writer.add_unk_token_id(1)
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(3)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)

    def tensor(name, value):
        writer.add_tensor(name, value.detach().cpu().contiguous().numpy().astype(np.float32,
                                                                                copy=False))

    tensor("token_embd.weight", weights["embedding.weight"])
    tensor("output_norm.weight", weights["norm.weight"])
    for index in range(config["n_layers"]):
        source, dest = f"blocks.{index}.", f"blk.{index}."
        for dst, src in (("attn_norm", "attention_norm"), ("ffn_norm", "ffn_norm"),
                         ("attn_output", "attention.proj"), ("ffn_gate", "gate"),
                         ("ffn_up", "up"), ("ffn_down", "down")):
            tensor(dest + dst + ".weight", weights[source + src + ".weight"])
        qkv = weights[source + "attention.qkv.weight"].chunk(3, dim=0)
        for name, weight in zip(("attn_q", "attn_k", "attn_v"), qkv, strict=True):
            tensor(dest + name + ".weight", weight)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Experimental float32 GGUF saved to {args.output}")


if __name__ == "__main__":
    main()
