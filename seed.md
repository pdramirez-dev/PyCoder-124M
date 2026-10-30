# Project: PyCoder-124M — Python Language Model from Scratch

Act as a senior AI/ML engineer specializing in PyTorch, Transformer architecture, LLM training from scratch, and Apple Silicon optimization.

Build **PyCoder-124M**, an autoregressive language model specialized in Python code generation and autocompletion, trained entirely from scratch.

## 1. Objective

Create a professional, modular, extensible project that can:

- Implement a decoder-only GPT architecture from scratch using PyTorch.
- Initialize all weights randomly, without pretrained models.
- Train a BPE tokenizer specialized in Python.
- Pretrain the model using next-token prediction.
- Generate and autocomplete Python code.
- Save checkpoints and resume interrupted training.
- Run evaluations and benchmarks.
- Scale the model later.

## 2. Hardware

Initial hardware:

- Apple MacBook Air M5.
- 32 GB of unified memory.
- macOS with Apple Silicon.
- Backend PyTorch MPS (Metal Performance Shaders).

The project must also support CUDA for later training on NVIDIA GPUs.

## 3. Model Architecture

Name: `PyCoder-124M`

Architecture: decoder-only GPT Transformer.

Proposed initial configuration:

- 12 Transformer blocks.
- Hidden dimension: 768.
- 12 attention heads.
- Context length: 1024 tokens.
- BPE tokenizer with configurable vocabulary.
- Causal self-attention.
- Rotary Position Embeddings (RoPE).
- RMSNorm.
- SwiGLU.
- Weight tying between input and output embeddings.
- Weight initialization from scratch.

Adjust internal dimensions and vocabulary to reach approximately 124 million parameters.

Implement automatic parameter counting and document the exact result.

Use `torch.nn.functional.scaled_dot_product_attention` where supported, with a functional fallback.

## 4. Technology Stack

- Python 3.12.
- PyTorch.
- Hugging Face Tokenizers.
- NumPy.
- PyYAML.
- tqdm.
- TensorBoard.
- pytest.
- Ruff.
- uv for dependency management.

Do not use `AutoModelForCausalLM` or load pretrained models.

## 5. Project Structure

```text
pycoder/
├── README.md
├── pyproject.toml
├── configs/
│   ├── model_124m.yaml
│   └── train.yaml
├── src/
│   └── pycoder/
│       ├── __init__.py
│       ├── config.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── attention.py
│       │   ├── transformer.py
│       │   ├── embeddings.py
│       │   └── gpt.py
│       ├── data/
│       │   ├── tokenizer.py
│       │   ├── dataset.py
│       │   └── preprocessing.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── checkpoint.py
│       │   └── optimizer.py
│       └── inference/
│           └── generate.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── tests/
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_training.py
├── data/
├── checkpoints/
└── logs/
```

## 6. Training

Implement a training loop with:

- AdamW optimizer.
- Learning rate warmup and cosine decay.
- Gradient accumulation.
- Gradient clipping.
- Periodic validation loss evaluation.
- Mixed precision when the backend supports it reliably.
- Periodic checkpointing.
- Full training resumption, including optimizer, scheduler, RNG, and dataset position.
- TensorBoard logging.
- Automatic selection between MPS, CUDA, and CPU.

Training must be stoppable with Ctrl+C and save a recovery checkpoint.

## 7. Dataset

In this first stage, do NOT download massive datasets.

Create a small demo dataset with original Python functions and valid code examples to test the pipeline.

The system must later support datasets such as The Stack v2, filtering for Python code and respecting licenses.

Split training and validation, avoiding duplicates and contamination between sets.

## 8. Validation

Implement automated tests covering:

- Tensor dimensions.
- Correct causal mask behavior.
- Forward pass.
- Backpropagation.
- Loss reduction on a small dataset.
- Autoregressive generation.
- Checkpoint saving and restoration.
- MPS operation, when available.

## 9. MacBook Air M5 Optimization

The MacBook Air has no fan, so the system must support controlled, resumable training.

Include:

- Configurable batch size and gradient accumulation.
- Configurable step limits.
- Memory monitoring where possible.
- A tiny model for quick tests.
- A separate production configuration for 124M.

Do not start long training runs automatically.

## 10. First Delivery

In this phase, only:

1. Create the complete scaffolding.
2. Implement the functional architecture.
3. Implement a tokenizer and minimal data pipeline.
4. Configure training and checkpoints.
5. Create the tests.
6. Document installation and commands.
7. Run a brief smoke test if the environment allows.

Do not yet run 100-million-token pretraining.

Prioritize clean, typed, modular, documented, maintainable code.

Do not create fake implementations, empty functions, or placeholders that appear complete.

On completion, show:

- Generated file structure.
- Exact parameter count.
- Test results.
- Smoke test result.
- Observed memory use, if measurable.
- Commands to run the project.

**Start implementing the project, rather than only describing how to do it.**
