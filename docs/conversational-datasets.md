# Candidate datasets for a conversational PyCoder

Research snapshot: 2026-10-31. Dataset cards and terms can change; pin revisions
and review licenses and provenance before downloading for an experiment.
These are candidates, not an assertion that their individual examples are safe,
correct, or cleared for every use.

## Candidates

| Dataset | Stage | Size / declared license | Notes |
| --- | --- | --- | --- |
| [Avelina/python-edu-cleaned](https://huggingface.co/datasets/Avelina/python-edu-cleaned) | Python pretraining | 7.68M rows; ODC-BY | Source of the sample used for the current model. Rows expose repository and path but not per-file license in the viewer; check provenance and overlap. |
| [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | Python instruction tuning | 50,661 rows; ~90 MB download; ODC-BY | Synthetic instructions and execution-filtered responses; execution filtering is not a correctness guarantee. Review source licensing and benchmark overlap. |
| [Nan-Do/instructional_code-search-net-python](https://huggingface.co/datasets/Nan-Do/instructional_code-search-net-python) | Python instruction tuning | 418,545 rows; ~173 MB download; Apache-2.0 declared | English instruction/response pairs generated from CodeSearchNet. Inspect source-code provenance and split by repository where possible. |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Conversation | Viewer has ~84,400 training messages; Apache-2.0 | Includes Spanish and English; rows are messages in conversation trees, not independent conversations. Reconstruct paths and split by tree. |
| [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Conversation | Entire collection ~87.7 GB; mixed sources | New subsets are declared Apache-2.0, existing subsets retain their own terms. Inspect multilingual subsets for usable Spanish examples; sample selectively, not the full collection. |

## Proposed experiment

1. Preserve the 124M checkpoint, tokenizer and validation data. Expanding to
   1B is a new architecture and cannot be resumed as if it were the same model.
2. First establish a reproducible functional and English-chat baseline with
   held-out prompts. Check dataset overlap, including approximate duplicates.
3. Review a small, pinned sample of BigCode instruction data and OASST1 English
   conversation paths. Record the actual usable row counts and inspect answer
   quality, duplicates, content origins and license obligations.
4. Build a separate chat preprocessing and supervised-fine-tuning pipeline. The
   existing Python corpus filters use `ast.parse` and cannot process chat. Define
   role delimiters and EOS behavior consistently for training and inference;
   choose a loss mask for assistant responses and preserve conversation boundaries.
   Changing the tokenizer vocabulary requires compatible embeddings and export.
5. Run a brief 124M trial with separate outputs, measure regression in code and
   conversation, and compare against the unchanged baseline. Only then consider
   larger samples, further pretraining or a separately designed 1B model.

The 36 GB Mac constrains model and optimizer memory; corpus files can be streamed
or processed in chunks. Neither dataset size nor a small SFT trial guarantees
Gemma-like capability. The current checkpoint was trained primarily on Python,
so natural-language and Spanish competence are not established.

## Pilot prepared on 2026-10-31

The source shards were downloaded at fixed revisions with SHA-256 hashes in
`data/raw/conversation-pilot/sources.json`. The data is local and ignored by Git.
`scripts/prepare_conversation_pilot.py` selects at most one item per BigCode seed
or OASST conversation tree, reconstructs reviewed Spanish paths, removes exact
normalized first-prompt duplicates, and deterministically separates groups.
The pilot has 799 training, 100 validation, and 100 test conversations. It does
not remove near duplicates or prove absence of benchmark contamination.

Only a handful of examples were inspected manually. One Spanish response
incorrectly states that the preterite "rio" takes an accent; it is evidence the
pilot is **not curated** and must not be treated as a quality-approved dataset.
Dataset labels and a review flag do not verify correctness; review licensing,
attribution and source content before further training or distribution.

`scripts/train_conversation_pilot.py` performs an experimental, fresh SFT trial
with a separate optimizer and a weights-only output; it cannot exactly resume an
interrupted trial. Role labels are ordinary text encoded using the original
tokenizer. The original experiments used `legacy-v1`, with loss on assistant text
and final conversation EOS only. New trials default to `turn-eos-v2`, which trains
an EOS immediately after every assistant reply and shares its formatter with
generation. Conversations longer
than the context window are excluded, not truncated. A three-step MPS plumbing
test ran from the 124M checkpoint and produced
`checkpoints/conversation-pilot-3step.pt`; metrics and artifact hashes are in
[conversation-pilot-3step.json](training-runs/mps-2026-10-31/conversation-pilot-3step.json).
The validation metrics use only ten examples per source and do not establish
improvement; the held-out test set was not used. A Spanish greeting prompt led
the three-step trial to repeat role labels instead of answering.

To reconstruct the pilot with available network access:

```sh
uv run python scripts/download_conversation_sources.py
uv run python scripts/prepare_conversation_pilot.py --language en \
  --output data/processed/conversation-pilot-en
```

Both commands refuse to overwrite existing output directories. The experiment
does not update the base checkpoint or tokenizer. For a fresh trial, use a new
output path, for example:

```sh
uv run python scripts/train_conversation_pilot.py checkpoints/latest.pt \
  --data data/processed/conversation-pilot-en \
  --output checkpoints/conversation-pilot-new.pt --device mps --steps 3
```

This is a technical workflow test, not an endorsed recipe for training on the
uncurated dataset. Before a longer experiment, build a reviewed sample, add a
fixed English-chat evaluation and compare against the base model.

## English pilot

The current pilot uses English OASST paths and the same BigCode source, with
799 training, 100 validation and 100 test examples. The 2026-10-31 three-step
MPS experiment is saved separately as `checkpoints/conversation-pilot-en-3step.pt`.
Its [metrics](training-runs/mps-2026-10-31/conversation-pilot-en-3step.json)
are a plumbing check, not evidence of conversational improvement. The English
data still requires quality, licensing and overlap review. The older Spanish
pilot remains separate for provenance and is not the default for new training.

## Broader mixed experiment

The next trial expanded to 785 structurally filtered conversations and alternated
200 SFT updates with 200 updates from the original Python training stream.
Sampled validation losses improved, but free-form conversation remained unreliable
and code scored 9/26 cases versus the base model's 13/26. Full setup, limitations,
results and reproduction commands are in the
[mixed-trial report](training-runs/mps-2026-10-31/diverse-en-400step.md).

A subsequent SFT-only arm matched the mixed trial's data, seed, learning rate and
400-update budget. It scored 6/26 code cases versus 9/26 for the mixture; neither
produced satisfactory answers on the fixed chat prompts. A small completion-choice
diagnostic found that the base already prefers many elementary English answers.
The [controlled comparison](training-runs/mps-2026-10-31/controlled-sft-comparison.md)
records the measurements, exposure differences and limits on interpretation.

The subsequent [format audit](training-runs/mps-2026-10-31/chat-format-review.md)
found no tokenizer-prefix mismatch but corrected EOS supervision for each
assistant turn. A v2 mixed pilot retained 9/26 code cases and did not show a
clear chat improvement. New checkpoints record their formatting version;
historical runs require `--chat-format legacy-v1` for reproduction.

An SFT-only v2 run increased exposure to 1,500 updates and selected step 1,300
by periodic validation. Both source losses decreased, but the fixed chat prompts
remained poor and code passed 4/26 cases. This is a warning against extending
SFT based on validation loss alone; see the [1,500-update report](training-runs/mps-2026-10-31/diverse-en-sft-1500step-v2.md).
