# Reviewed English format-learning experiment

A bounded trial used 11 explicitly selected training examples (six BigCode code
responses and five short OASST English responses). Selection rationales are in
`configs/conversation_review_en.json`. Selection was performed by the coding
agent, not independent human annotation. All examples came from the training
split; validation and test files were retained unchanged. Six selected Python
functions passed 15 small cases in restricted Docker containers; this does not
prove correctness for every input or clear all source-license obligations.

Starting from the original 124M checkpoint, 100 updates ran in float32 on MPS,
batch size 1, learning rate 0.00005. This revisited the eleven examples about nine
times. The output is `checkpoints/conversation-reviewed-en-100step.pt`. The
base checkpoint, tokenizer and local Ollama model were not replaced.

## Observations

| Metric | Base | After 100 updates |
| --- | ---: | ---: |
| BigCode reference loss (10 validation examples) | 2.1461 | 2.1673 |
| OASST English reference loss (10 validation examples) | 4.9733 | 5.5272 |
| Existing signature-only code cases passed | 13/26 | 0/26 |

Training loss became very low, while both sampled validation losses increased.
This is consistent with overfitting to the tiny training set. The code comparison
uses the same six prompts, 100-token limit, greedy decoding and first-function
extraction as the original signature-only evaluation. After tuning, all six
first functions failed parsing or execution; the original model passed all
cases only for addition. The code suite was used previously during development,
so these comparisons are diagnostics, not a fresh unbiased test benchmark.

In the six fixed English chat prompts, the tuned model produced nonempty replies
and reached EOS for five prompts; one reached the token limit with repetition.
The greeting became relevant, although it did not obey the one-sentence request.
Other responses reused training phrases unrelated to the question. This shows
format acquisition without reliable instruction following or generalization.
It does not meet the roadmap's acceptance criterion.

## Decision and next experiment

Retain the base as the usable code model and keep this checkpoint for analysis.
Do not scale this eleven-example recipe to 1B. Next, increase the diversity and
review coverage of the instruction data, establish a separate development set
for selecting schedules, and test mixing code-completion rehearsal with chat
examples to measure whether it reduces regression. A new final evaluation set
will be needed after these development prompts have influenced choices.

Artifacts: [training metrics and hashes](conversation-reviewed-en-100step.json),
[chat outputs](chat-reviewed-en-100step.json),
[code outputs](code-reviewed-en-100step-samples.json),
[isolated code scores](code-reviewed-en-100step-scores.json),
and [selected code-example checks](reviewed-code-checks.json).

Reproduce the selected subset using `scripts/prepare_reviewed_pilot.py` with the
original pilot and review configuration. For a new training run use a new output
filename, and pass `--data data/processed/conversation-reviewed-en`, `--steps 100`
and `--learning-rate 0.00005 --chat-format legacy-v1` to `scripts/train_conversation_pilot.py`.
