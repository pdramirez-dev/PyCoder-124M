# English chat baseline and data review

The fixed suite in `configs/chat_eval_en.json` contains four short English
conversation prompts and two code requests. It is a diagnostic requiring human
review, not a validated benchmark or an automatic correctness score. The
runner uses greedy decoding, a maximum of 80 new tokens and stops at a generated
role delimiter or EOS. It records output, checkpoint/tokenizer/suite hashes,
device and generation settings. A role stop can leave an empty answer.

The base and three-step checkpoints were evaluated with the same prompts and
settings. Complete responses are available in [base results](chat-baseline-en.json)
and [three-step results](chat-pilot-en-3step.json). The base produced empty
responses for three of six prompts, irrelevant responses for two, and a
repetitive response for one. The three-step checkpoint produced empty responses
for all six, stopping at the next role marker. These results do not show a
conversation improvement. Three optimizer updates are a plumbing test, not a
meaningful instruction-tuning experiment.

No exact normalized prompt match or exact eight-word sequence match was found
against the first user prompt in the 999 English pilot examples. This does not
check the original pretraining corpus, approximate paraphrases, assistant text,
or other outside sources and therefore does not establish contamination freedom.

## Initial quality review

Inspected seven examples from the beginning of the training selection. This is a
small convenience sample, not a statistically representative audit:

- BigCode `560272`: linear search response matches the stated example; no
  independent execution was performed during this manual review.
- BigCode `113402`: code implements a recursive copy of nested dictionaries;
  the meaning of “tree-like” is underspecified, so suitability is uncertain.
- BigCode `157560`: the prompt describes a list of named values, while the
  response uses a mapping's `.get`; the data contract is ambiguous.
- BigCode `220044`: the example works as described, but no behavior is defined
  for lists shorter than two items and the implementation raises `IndexError`.
- OASST `c73f7111-397a-4135-9e2e-48f937882209`: response conflates a series
  of wars with a single battle and is misleading in its historical framing.
- OASST `c8800cae-3cb4-4337-9254-ac096c83832b`: assistant declines the prompt
  rather than addressing it; it is not a useful target for helpful dialogue.
- OASST `cb79854a-518c-47aa-aa34-370e95325a50`: explanation of PGO is broadly
  relevant, but its implementation details and claimed effort are generalized.

Neither the reviewed subset nor the rest of this dataset is approved as a
quality-controlled training set. Longer training is deferred until a reviewed
subset and a held-out evaluation protocol exist. Existing checkpoints are not
overwritten.

To reproduce the chat samples using a trusted checkpoint and unchanged tokenizer:

```sh
uv run python scripts/evaluate_chat_samples.py checkpoints/latest.pt \
  /tmp/pycoder-chat-eval.json --device mps
```

The script only generates text; do not execute model-generated code on the host.
Use isolated execution for functional verification.
