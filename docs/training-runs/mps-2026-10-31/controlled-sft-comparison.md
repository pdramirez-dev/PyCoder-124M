# Controlled update-budget comparison and English diagnostic

The new SFT-only experiment starts from the same base checkpoint and uses the
same dataset, tokenizer, validation file, seed, learning rate (0.00002), batch
size (1), device (MPS) and total update count (400) as the prior mixed run. These
values and artifact hashes were checked against both reports.

The intervention is the training composition: mixed uses 200 SFT updates and
200 updates on the original Python stream (context 256); SFT-only uses 400 SFT
updates. Thus the total updates match, but SFT exposures and input token counts
do not. The mixed run processed approximately 110,134 supervised target tokens
(58,934 SFT plus 51,200 code); SFT-only processed 108,015 supervised target tokens.
Their input token counts were 131,011 and 152,558, respectively. Different
sequence lengths, masks and gradient histories still prevent a strict
compute-matched or token-matched causal claim.

## Results

| Diagnostic | Base | Mixed | SFT only |
| --- | ---: | ---: | ---: |
| BigCode sampled validation loss | 2.1461 | 1.7658 | 1.6877 |
| OASST English sampled validation loss | 4.9733 | 4.8610 | 4.8159 |
| Existing signature-only code cases | 13/26 | 9/26 | 6/26 |
| Functions passing every case | 1/6 | 1/6 | 1/6 |
| English choices, mean token NLL | 10/12 | 9/12 | 10/12 |
| English choices, summed NLL | 9/12 | 8/12 | 9/12 |

The fixed six-prompt chat review found no satisfactory response in either broader
adjustment: replies were irrelevant, repetitive, or incorrect code. Both adjusted
models improved sampled teacher-forced validation loss while failing to become
useful conversational assistants. Code rehearsal reduced the observed regression
relative to this SFT-only run, but the difference is three cases in a small suite
and should not be treated as a general or statistically reliable improvement.

## What the English diagnostic does and does not establish

`configs/english_choice_probe.json` has twelve authored pairs of completion
choices about basic vocabulary, grammar, categories, quantity and short context.
The script scores only continuation tokens, reporting both mean and summed
negative log-likelihood; lower is preferred. It never executes candidate code.
The summed score favors shorter tokenizations, while mean normalization can
also change option preferences. Token counts are retained for inspection.

The base prefers the expected option for many of these examples, so the
conversational failure cannot be attributed solely to an inability to model
simple English. This is a tiny, unvalidated diagnostic with two choices per item,
not a comprehension benchmark, and does not measure language proficiency or
rule out memorization. No overlap claim is made against the full pretraining
corpus. Single-seed experiments cannot isolate all sources of variation.

The experimental checkpoints are retained separately and neither replaces the
base nor the Ollama deployment. Next priority is to test the SFT recipe and
training/generation agreement with a larger reviewed instruction set and a
separate development suite; increasing to 1B is not justified by these results.

Artifacts:

- [SFT-only training report](conversation-diverse-en-sftonly-400step.json)
- [SFT-only chat output](chat-diverse-en-sftonly-400step.json)
- [SFT-only code samples](code-diverse-en-sftonly-400step-samples.json)
- [SFT-only isolated scores](code-diverse-en-sftonly-400step-scores.json)
- English choices: [base](english-choice-base.json), [mixed](english-choice-mixed.json),
  [SFT-only](english-choice-sftonly.json).

To reproduce the SFT-only arm with new output files:

```sh
uv run python scripts/train_conversation_pilot.py checkpoints/latest.pt \
  --data data/processed/conversation-diverse-en --device mps \
  --output checkpoints/sftonly-new.pt --steps 400 --learning-rate 0.00002 \
  --chat-format legacy-v1
uv run python scripts/evaluate_english_choices.py checkpoints/sftonly-new.pt \
  /tmp/english-probe-new.json --device mps
```
