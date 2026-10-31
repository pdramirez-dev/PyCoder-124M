# PyCoder-124M

See the [Training Guide](docs/training.md) for hardware profiles, data preparation,
CUDA/MPS commands, monitoring, checkpoint recovery, and troubleshooting.

An autoregressive Python code model initialized from scratch. The implementation uses a decoder-only Transformer with RoPE, RMSNorm, SwiGLU, and tied embeddings, without pretrained weights. The production configuration has **123,587,328 parameters**: 12 blocks, dimension 768, 12 heads, FFN 2048, vocabulary 50,304, and context 1024. `configs/model_tiny.yaml` is for testing, not measuring final model quality.

## Requirements and Installation

Python 3.12, [uv](https://docs.astral.sh/uv/), and macOS, Linux, or Windows with a compatible PyTorch version are required. CUDA takes priority, MPS is selected on Apple Silicon when available, and CPU is the fallback. Install dependencies and development tools:

```sh
uv sync
```

## Quick Demo

`data/demo` contains small original examples; it does not produce a useful model. The following commands train the tokenizer only on training files, split validation before tokenization, and run just three steps:

```sh
uv run python scripts/train_tokenizer.py --vocab-size 512
uv run python scripts/prepare_dataset.py
uv run python scripts/train.py --config configs/train_smoke.yaml
uv run python scripts/evaluate.py checkpoints/latest.pt
uv run python scripts/generate.py checkpoints/latest.pt --prompt 'def '
uv run pytest -q
uv run ruff check src scripts tests
```

For configurable training, edit `configs/train.yaml` and run:

```sh
uv run python scripts/train.py --model configs/model_tiny.yaml --config configs/train.yaml
uv run tensorboard --logdir logs
```

For 124M, first prepare a sufficient corpus (see "Real Corpus") and a tokenizer with `--vocab-size 50304`. `configs/train_124m.yaml` trains on about 100M tokens: 1,526 steps of 65,536 tokens (batch 4 × accumulation 16 × context 1024), approximately 12 hours on a MacBook Air M5 in float32 before thermal throttling. Each checkpoint occupies about 1.5 GB.

```sh
uv run python scripts/train.py --model configs/model_124m.yaml --config configs/train_124m.yaml
```

For an 8 GB NVIDIA GPU, `./run.sh --device cuda` selects
`configs/train_124m_cuda.yaml`: bfloat16, batch size 1, and accumulation 64.
This preserves 65,536 tokens per optimizer step. `./run.sh --device mps`
uses the original float32 configuration. An explicit `--config` overrides these defaults.
Checkpoints require identical model and training configurations when resuming;
an existing float32 checkpoint cannot resume with the new CUDA configuration.
Start the CUDA run with a separate checkpoint directory, for example:

```sh
./run.sh --device cuda --checkpoints checkpoints/cuda --logs logs/cuda
# Resume a checkpoint created with this configuration:
./run.sh --device cuda --resume checkpoints/cuda/latest.pt --checkpoints checkpoints/cuda --logs logs/cuda
```

To inspect hardware and suggest a profile without starting training:

```sh
uv run python -m pycoder.hardware  # Auto-detect CUDA, MPS, or CPU
./run.sh --suggest-profile --device cuda
./run.sh --suggest-profile --device cuda --output configs/train_local.yaml
./run.sh --device cuda --config configs/train_local.yaml --checkpoints checkpoints/local
```

The advisor reports system RAM and, on CUDA, total/free VRAM. It selects a conservative
batch using free VRAM, prefers bfloat16 when supported (otherwise float16 on CUDA),
and preserves 65,536 tokens per step through gradient accumulation. MPS and CPU use
float32. Suggestions apply only to the 124M model with context length 1024 and are
heuristics, not measured memory guarantees. System RAM is separate from CUDA VRAM;
MPS uses unified memory. RAM figures reflect what the current environment exposes.
The advisor never starts training or overwrites an existing YAML file. Use the
original training configuration to resume an existing checkpoint.

Training parameters, batch size, and accumulation are configurable. The demo corpus is insufficient for the production context. Long training runs are not started automatically.

## Real Corpus

`docs/datasets.md` lists the Hugging Face datasets evaluated. To download a sample of `Avelina/python-edu-cleaned` (by default 400 MB of text, about 100M tokens) into `data/raw/`, with provenance (revision and shards) in `data/raw/python-edu.json`:

```sh
uv run python scripts/download_dataset.py --max-mb 400
uv run python scripts/train_tokenizer.py --source data/raw --vocab-size 50304 --validation-fraction 0.05
uv run python scripts/prepare_dataset.py --source data/raw --validation-fraction 0.05
```

`train_tokenizer.py` and `prepare_dataset.py` read `.py` files and rows from `.parquet` files (`--text-column`, default `text`). The sample takes rows in shard order, deterministically for the same revision.

Both scripts apply the basic filter in `configs/filters.yaml` (`--filters`) before deduplication and splitting, so the tokenizer and dataset see the same files. They discard non-UTF-8 documents, documents that are too small or large, have very long lines or a low alphanumeric fraction, are marked as autogenerated in their first lines, or fail `ast.parse` (including Python 2). A leading BOM is removed. `manifest.json` records the filter configuration and rejection counts by reason. Use the same filter file in both scripts. There is no approximate deduplication, decontamination, or per-file license filtering.

## Checkpoints and Resuming

The latest checkpoint is written atomically to `checkpoints/latest.pt`; it includes weights, optimizer, scheduler, scaler, RNG states, and training stream position. Ctrl+C requests a stop and save after the current step completes. Resume with the same files, configuration, and backend:

```sh
uv run python scripts/train.py --config configs/train.yaml --resume checkpoints/latest.pt
```

The configuration and training-file hash are checked when resuming. Keep the original tokenizer and split as well; replacing them may invalidate evaluation or generation. For exact reproducibility, use the same hardware, PyTorch version, and backend; some accelerator operations may be nondeterministic. Checkpoints contain Python objects: load only trusted files.

## Data and Precision

`prepare_dataset.py` accepts a directory containing `.py` files. It deduplicates by exact content hash and splits whole files deterministically. It does not detect approximate duplicates or code shared between files; a real corpus needs additional filtering, contamination analysis, and license and provenance review before training. If `--seed` or `--validation-fraction` changes, use the same values in `train_tokenizer.py` and `prepare_dataset.py`. The BPE vocabulary can be smaller than the requested maximum for a small corpus; generation restricts sampling to the effective vocabulary.

The stream concatenates documents delimited by BOS/EOS and uses contiguous cyclic windows. Attention across documents within a window is allowed. Validation loss and perplexity reflect this stream and are not benchmarks of functional code correctness. The Stack v2 integration and functional benchmarks are planned for a later stage.

`precision: float32` is the default and works with CPU/MPS/CUDA. `float16` and `bfloat16` are enabled only on CUDA; verify device support. TensorBoard records allocated MPS or CUDA memory when available. Total system unified memory may differ from that measurement. There is no automatic thermal throttling; adjust steps, batch size, and breaks for a fanless MacBook Air.
