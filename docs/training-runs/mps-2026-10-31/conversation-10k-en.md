# 10,000-example English synthetic pilot

The local dataset `data/processed/conversation-10k-en` has 10,000 training
conversations: 23 previously inspected external examples and 9,977 generated
examples. The synthetic portion has ten families, with 997 or 998 examples each:
addition, subtraction, multiplication, uppercase, lowercase, length, reversal,
sorting, parity and Python functions adding a constant. String operations use
invented lowercase letter sequences, not natural-language conversations.

Answers are generated deterministically with Python from seed 3101. Tests
independently check the arithmetic, strings, sorting, parity and the parsed
structure and constant of every synthetic example produced with that seed.
The actual output has 10,000 examples within the original tokenizer's 1,024-token
context; the longest encoded example is 295 tokens. The generated records are
labeled `synthetic-en`, with family and stable source/group IDs. The manifest
records source and generator hashes, counts, answer frequencies, and hashes of
the development suites.

The generator excludes normalized exact prompt matches with reviewed training
examples, every user turn in the held-out splits, and the two development chat
suites. It preserves existing validation and test bytes. Exact comparison does
not establish absence of paraphrase, task-family, or pretraining overlap. The
test split is not used to select a checkpoint.

This is template-generated data, not 10,000 reviewed or independently written
conversations. It is 99.77% synthetic and heavily weighted toward short answers;
the parity family alone contains 528 `No` and 469 `Yes` answers. The 997 code
examples are variants of one simple function pattern. These properties can
encourage repetition and cannot establish conversational competence. Original
external examples retain their single-reviewer and license/provenance limits.
No training run is implied by creating this dataset.

Recreate from the existing reviewed source in a new output directory:

```sh
uv run python scripts/prepare_10k_pilot.py \
  --source data/processed/conversation-reviewed-23-en \
  --output data/processed/conversation-10k-new --seed 3101
```

The source dataset can itself be rebuilt following the 23-example trial report.
Generated data and checkpoints are ignored by Git. The generator refuses to
overwrite an existing output directory.
