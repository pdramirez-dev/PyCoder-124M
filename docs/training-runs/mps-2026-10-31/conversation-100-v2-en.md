# Revised 100-example pilot

`data/processed/conversation-100-v2-en` keeps the same 23 reviewed external
conversations and replaces the original 77 synthetic examples with 77 new
examples: seven each in eleven task families. The families include arithmetic,
letter case, word length, reversal, sorting, parity, repetition and first
letter. Seven distinct word lengths replace the original all-five-letter set.
Length answers use both numeric-only and sentence formats; parity has both yes
and no answers. The most common exact synthetic answer occurs four times,
compared with eleven occurrences of `5` in the earlier synthetic set.

This removes a visible answer imbalance but does not establish that model
collapse is fixed. The synthetic examples still use a small number of templates,
repeat the same seven words across tasks and do not provide broad conversation
or code coverage. The 23 external examples retain their earlier review and
provenance limitations. The original dataset and experiment remain available.

The generator rejects duplicate normalized synthetic prompts and exact matches
against reviewed training prompts, all user turns in the held-out splits, and
the development chat prompts. It preserves validation and test bytes and stores
input hashes and answer frequencies in its manifest. Exact comparisons do not
rule out paraphrase overlap or pretraining contamination. The held-out test
split was not used for checkpoint selection.

Recreate in a new output directory after preparing the 23-example source:

```sh
uv run python scripts/prepare_100_pilot_v2.py \
  --source data/processed/conversation-reviewed-23-en \
  --output data/processed/conversation-100-v2-new
```

For a controlled comparison, use the same base checkpoint, tokenizer, update
schedule, seed, validation subset, code rehearsal and evaluation prompts as the
[first 100-example experiment](conversation-100-mixed-200step.md). Record both
the chosen step and data exposure. Measure response diversity and task accuracy
on independent examples, including unseen values, before concluding that v2
improves generalization.
