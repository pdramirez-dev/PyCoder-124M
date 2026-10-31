# Running PyCoder in Ollama or LM Studio on Mac

An experimental float32 GGUF export and an Ollama import were created on
October 31, 2026. The training checkpoint remains unchanged. The conversion is
promising but not yet exhaustively validated.

## Experimental export and local Ollama model

For a trusted checkpoint and its original tokenizer, from the project root:

```sh
uv run --with gguf python scripts/export_gguf.py checkpoints/latest.pt \
  checkpoints/pycoder-f32.gguf
ollama create pycoder-124m-experimental -f configs/Modelfile.ollama
ollama run pycoder-124m-experimental 'def add(a, b):'
```

The exporter refuses to overwrite an existing output. Its optional `--tokenizer`
argument must point to the exact tokenizer used during training; matching
vocabulary size alone cannot establish tokenizer identity. The GGUF is a local
generated file and is ignored by Git. The Ollama model is a separate local copy.
The Modelfile limits output to 100 tokens and configures context length 1024.
This is a code-completion model and does not have instruction-following training.

The experimental float32 file was approximately 473 MiB. With llama.cpp commit
`0ec4027a2a0183de87c007069adddc06e665f7de`, three CPU logit comparisons against
PyTorch gave maximum absolute differences of 0.00224–0.00286 and identical
highest-scoring tokens. Ten tokenizer examples matched, including whitespace,
Unicode and literal special tokens. One short Metal generation and an Ollama
generation succeeded. These limited checks do not prove full parity, especially
for long contexts or every possible input. On load, llama.cpp warned that the
EOS token was not already in its detected end-of-generation set, then added it;
end-of-generation behavior needs further validation. No quantization has been
tested. LM Studio import has not been tested.

## Candidate route

The experimental converter targets Llama GGUF. PyCoder has similarities
(RMSNorm, SwiGLU, RoPE and tied embeddings), but similarities alone do not
establish equivalent behavior.

The converter and verification need to account for:

- Fused Q/K/V weights in PyCoder versus separate weights in candidate backends.
- RoPE layout: PyCoder rotates interleaved pairs; verify whether Q/K weight
  permutation or another layout adjustment is required by the target runtime.
- RMSNorm epsilon: PyCoder uses PyTorch's default `eps=None`, so parity must
  use the actual float32 epsilon; precision changes can affect this behavior.
- The byte-level BPE tokenizer, its pre-tokenization, merges, vocabulary,
  special-token IDs and BOS/EOS behavior. Compare token IDs on whitespace,
  indentation, Unicode and code samples before comparing model outputs.
- Context length 1024, causal masking, tied embeddings and inference behavior
  at the context boundary.

First compare fixed-prompt logits between PyTorch and an unquantized float32
export within a justified numerical tolerance, plus greedy generation and
tokenizer IDs. Only after parity is established, measure float16 and quantized
variants (for example Q8 and Q4) for memory, throughput and functional quality.
Keep the original checkpoint and tokenizer unchanged. An MLX implementation is
another possible Apple Silicon route, but its compatibility with LM Studio's
supported model formats and this architecture also requires verification.

Ollama is installed on the Mac checked during this investigation. LM Studio
has not been verified. Ollama can import compatible
GGUF files through a Modelfile; LM Studio can import compatible GGUF files via
its CLI. Both require a model their runtime can execute. PyCoder is a code
completion model, not an instruction-tuned chat assistant.

References: [llama.cpp model architecture guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md),
[Ollama import](https://docs.ollama.com/import),
[LM Studio model import](https://lmstudio.ai/docs/app/advanced/import-model),
and [MLX Llama implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama.py).
