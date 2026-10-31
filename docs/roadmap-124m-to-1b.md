# PyCoder: development plan from 124M to 1B

The current goal is an experimental English-language programming assistant, inspired by
the staged development of Gemma 3. This is not a reproduction of Gemma's
architecture, data, multimodal capabilities, training scale or performance.
The existing 124M checkpoint is the baseline; a 1B architecture requires a new
model and cannot directly resume the 124M checkpoint.

## 0. Preserve the baseline and control generation

- Keep the checkpoint, tokenizer, corpus manifest, validation split, configuration
  and existing GGUF. Never regenerate these in place.
- Record code-completion and English-conversation baselines with fixed prompts,
  deterministic settings and held-out tests. Track exact checkpoint/tokenizer
  hashes, environment, hardware and experiment settings.
- Limit generation and support explicit stop sequences. For single-function code
  completion, stop before the next top-level function or class; this is a
  heuristic, not proof that the first function is correct or complete.
- Acceptance: existing functional suite remains repeatable; stop sequences work
  across token boundaries and do not trigger on the supplied prompt.

## 1. Curate data and evaluation

- Inspect pinned samples of BigCode Self-OSS-Instruct and English OASST1; see
  [dataset research](conversational-datasets.md). Review licenses, provenance,
  quality and suitability before use; record revisions and rejection statistics.
- Split by source repository or conversation tree where available, deduplicate
  across splits and look for benchmark overlap. Maintain a separate held-out
  evaluation set, including English requests and functional Python cases.
- Create a separate chat dataset pipeline that reconstructs conversation paths,
  formats roles and masks user/padding tokens in the assistant-response loss.
  Decide role delimiters and EOS behavior consistently across training and
  inference. If new special tokens are added, initialize/resize embeddings and
  update export; do not silently replace the tokenizer of an existing checkpoint.
- Acceptance: stable manifests, reproducible splits, valid examples and tests for
  masking, truncation, end-of-turn behavior and no validation leakage.

## 2. Trial instruction tuning on 124M

- Start from a copy of the 124M weights as an explicit new fine-tuning run with
  separate optimizer, schedule, checkpoints and logs. This is not an exact resume.
- Begin with a small reviewed set of code instructions and English conversations;
  compare training mixtures and the unchanged base under the same evaluation.
- Track code task pass rate, response format and English responses, as well as
  validation loss. Inspect regressions and potential memorization.
- Acceptance: held-out instruction following improves without unacceptable
  degradation on the existing code suite. Set thresholds after measuring the
  baseline rather than inventing targets.

## 3. Design and measure a 1B model

- Create a separate roughly 1B decoder configuration; calculate actual parameter
  count including tied embeddings. Consider attention, normalization and context
  decisions only after a baseline comparison. Gemma 3 is a reference, not a
  claim of architecture compatibility.
- Test initialization, one forward/backward/optimizer update, evaluation and
  checkpoint save/reload with representative context and batch. Measure peak
  unified memory, swap, throughput and disk use on the 36 GB Mac Studio.
- Our MPS path currently uses float32. If needed, evaluate smaller microbatches,
  accumulation and activation checkpointing, with correctness tests before
  adopting them. GPU availability on the Lenovo requires its own benchmark.
- Acceptance: repeated short runs fit without excessive memory pressure, save
  and resume correctly, and have measured cost. No long training until then.

## 4. Pretrain and tune the selected model

- Build a larger reviewed text/code corpus; keep a fixed validation split and
  explicitly quantify actual token counts, licenses, overlap and compute cost.
  Reusing the same small sample for many epochs is not a substitute for diverse data.
- Choose the tokenizer before training a new model; document English and code
  token efficiency. Preserve training artifacts and provenance.
- Start a bounded pilot, measure loss, throughput and held-out task results, then
  decide the training budget. Checkpoints and explicit resume behavior must be
  verified before extending it. Apply instruction tuning only after pretraining.
- Acceptance: demonstrate improvement over the 124M baseline on the same
  untouched evaluation set and document runtime, costs and limitations.

## 5. Export, parity and local serving

- Convert the chosen checkpoint to GGUF and compare tokenizer IDs and logits with
  PyTorch, including whitespace, Unicode, EOS and context-boundary cases.
- Compare float32, float16 and quantized variants on Metal for accuracy, memory
  and speed. Validate stop behavior and the exact conversation template end to end.
- Run functional tests in restricted containers, then test Ollama and LM Studio
  independently. Report the results and versions; quantization is not assumed
  to preserve model quality.

No date or training-time estimate is committed until the 1B hardware pilot and
the actual dataset/token counts are measured. Do not publish checkpoints or data
without reviewing their licenses and provenance.
