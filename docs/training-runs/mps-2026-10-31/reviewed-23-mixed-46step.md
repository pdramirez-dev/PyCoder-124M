# Reviewed 23-example mixed SFT trial

Five more examples were manually selected from a short-example sample, for 23
training conversations (11 BigCode, 12 OASST). The selection and review reasons
are in [the approval list](../../../configs/conversation_review_23_en.json).
This is a small, single-reviewer selection, not a representative quality audit
or a license clearance. Short-answer sampling favors simpler examples.

The fresh run started from the base checkpoint with float32 MPS, batch size 1,
learning rate 2e-5 and turn-eos-v2 format. It alternated 23 SFT updates, visiting
each example once, with 23 updates on original code tokens (context 256). The
lowest average of the two validation source losses occurred at step 46.
On the first 20 usable examples per source, BigCode loss changed 2.0600 to
1.8642 and OASST loss changed 4.7462 to 4.7215. These are teacher-forced losses,
not measures of response correctness.

Manual review of the six fixed development chat prompts found no satisfactory
responses: five reached the 80-token cap, one EOS, and outputs included
irrelevant or repeated text. The same isolated signature-body code evaluation
used in earlier reports passed 11/26 cases: addition 4/4, evenness 2/5,
factorial 1/4, reverse 0/4, palindrome 2/4 and Fibonacci 2/5. The original base
passed 13/26 and the prior v2 mixed 400-update run 9/26. These runs differ in
data and update budgets, so the scores do not isolate the effect of this dataset.
The six prompts are repeatedly used development checks, not a final benchmark.

The result does not establish conversational ability or justify a longer run
on these 23 examples. Continue reviewing a substantially larger and more varied
training set and improve held-out evaluation before scaling this recipe. Source
provenance and licensing obligations remain to be reviewed.

Artifacts: [training metadata](conversation-reviewed-23-mixed-46step.json),
[chat responses](chat-reviewed-23-mixed-46step.json),
[code samples](code-reviewed-23-mixed-46step-samples.json),
[isolated scores](code-reviewed-23-mixed-46step-scores.json). The checkpoint
`checkpoints/conversation-reviewed-23-mixed-46step.pt` is local and weights-only.

```sh
uv run python scripts/prepare_reviewed_pilot.py \
  --source data/processed/conversation-diverse-en \
  --review configs/conversation_review_23_en.json \
  --output data/processed/conversation-reviewed-23-new
uv run python scripts/train_conversation_pilot.py checkpoints/latest.pt \
  --data data/processed/conversation-reviewed-23-new \
  --output checkpoints/conversation-reviewed-23-new.pt --device mps \
  --steps 46 --eval-every 23 --learning-rate 2e-5 \
  --rehearsal data/processed/train.bin --rehearsal-context 256
```
