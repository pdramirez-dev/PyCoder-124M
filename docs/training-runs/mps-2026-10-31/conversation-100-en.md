# 100-example English pilot

`data/processed/conversation-100-en` combines the 23 previously inspected
training conversations (11 BigCode, 12 OASST) with 77 locally generated
examples. The synthetic examples use seven task families with eleven instances
each: addition, subtraction, uppercase, lowercase, word length, reversal and
sorting. Answers are computed by the generator. They are short format-learning
examples, not 77 independently written conversations or a broad quality review.
The prior external examples have only single-reviewer inspection, and their
source provenance and licensing obligations still need review.

The generator checks for normalized exact prompt collisions against the original
training conversations, both held-out splits and the fixed development chat
suite. This does not identify paraphrases or overlap with pretraining data. It
preserves the validation and test files byte for byte and records source and
output hashes. Their contents were not used to construct answers. The test split
is not used for checkpoint selection.

The data is a small experimental pilot. Most rows share seven templates, so
overfitting and weak transfer are likely; reaching 100 rows does not establish
readiness for a general chat training run. Check generated samples and the same
code and chat evaluations after any experiment, while keeping a separate final
evaluation for generalization.

Recreate in a new output directory:

```sh
uv run python scripts/prepare_reviewed_pilot.py \
  --source data/processed/conversation-diverse-en \
  --review configs/conversation_review_23_en.json \
  --output data/processed/conversation-reviewed-23-new
uv run python scripts/prepare_100_pilot.py \
  --source data/processed/conversation-reviewed-23-new \
  --output data/processed/conversation-100-new
```

All generated datasets are local and ignored by Git. Preparation refuses to
overwrite an existing output directory.
