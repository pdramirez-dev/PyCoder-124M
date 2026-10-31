# Exploratory instruction data review

We inspected 16 training conversations from the 785-row diverse English set:
eight BigCode and eight OASST, sampled with `random.Random(42)` in source order.
The source file SHA-256, IDs, decisions, and reasons are recorded in
[`conversation_sample_review_en.json`](../../../configs/conversation_sample_review_en.json).
The sample contains 4 conditional accepts, 6 exclusions and 6 requiring further
review. These counts are exploratory and do not estimate the quality of the
entire dataset. Decisions reflect one reviewer and are not license clearance.

The findings explain why syntax filtering alone is insufficient: a cluster
function has the wrong signature, a title-case function contradicts the specified
capitalization, and a PHP conversation confuses a timezone argument with a date
format. Another training prompt embeds the exact solution copied into its
answer. These records remained in the prior training set. Their presence is a
data quality concern, but this sample does not prove they caused the model's
observed failures.

The six existing chat prompts cover greeting, clarification, explanation,
format following and two code tasks. They were repeatedly used to guide
development, have manual review criteria and only cover single-turn inputs.
They cannot establish general chat quality or serve as an untouched final test.
The separate dataset test split remains untouched; the original pretraining
corpus and paraphrase overlap have not been ruled out.

Before the next training run, expand independent review by source and task,
include executable checks for suitable Python responses, decide how to handle
ambiguous prompts and prompts containing solutions, and create a separate
multi-turn evaluation with explicit criteria. Keep the development prompts for
comparisons and reserve an independently constructed set for final assessment.
The four accepted examples alone are too few for a useful training corpus; the
review file is an audit record, not an automatic training dataset.

## Expanded selection

A further sample favored short single-turn examples (seed 183, 15 per source,
with overlap with previous reviews); it is deliberately not representative of
the full dataset. For example, BigCode `582354` takes middle positions without
ordering the values, `25552` returns the wrong type and function name, and
`319781` omits keyword arguments and cannot cache unhashable arguments. These
were not selected. The expanded approval list records individual reasons in
[`conversation_review_expanded_en.json`](../../../configs/conversation_review_expanded_en.json).

The prepared `data/processed/conversation-reviewed-expanded-v2-en` contains 18
examples (8 BigCode and 10 OASST), up from 11, with the original validation and
test files preserved. The preparation script records input hashes, rejects
duplicate selected IDs, and checks that selected source groups do not appear
in held-out splits. Exact group separation does not rule out paraphrase or
pretraining contamination. The examples are single-reviewer judgments, and
the two new code examples have not received exhaustive execution checks.

This is suitable only for a small format-learning experiment. It is not yet a
large enough curated corpus for a meaningful conversational retraining run;
repeated training on 11 examples previously overfit. Source license and
provenance review remain necessary. Recreate the selection with:

```sh
uv run python scripts/prepare_reviewed_pilot.py \
  --source data/processed/conversation-diverse-en \
  --review configs/conversation_review_expanded_en.json \
  --output data/processed/conversation-reviewed-expanded-new
```
