# Training/generation agreement and turn termination

The audit checked the actual tokenizer on all 785 conversations (1,129 assistant
turns) in the broader English training file. No token-prefix mismatch was found
in the original format. This rules out that particular mismatch for this sample;
it is not a proof for arbitrary inputs.

Two termination issues were identified in `legacy-v1`:

- Only the final conversation EOS was supervised. Intermediate assistant turns
  did not have an explicit end-of-turn target.
- Final EOS was conditioned on an appended newline, but the newline itself lay
  outside the assistant loss span. Generation was not taught that intervening
  token by this objective, making termination indirect.

`turn-eos-v2` uses a shared formatter for training and generation and places a
supervised EOS directly after every assistant response. User text and role
headers remain masked. The original tokenizer and vocabulary are retained;
role markers remain ordinary text and the existing EOS ID is reused. Padding
targets remain ignored. The complete audit found 1,129 supervised EOS targets
and zero generation-prefix mismatches with v2, versus 785 EOS targets in v1.
[Audit counts](chat-format-audit.json) include conversations that would be
excluded from training for length.

The trainer records the format in its checkpoint; the evaluator automatically
uses that metadata and falls back to legacy-v1 for older checkpoints. To
reproduce historical runs pass `--chat-format legacy-v1` explicitly. Tests check
loss masking, padding, intermediate EOS, token-prefix agreement, and metadata
selection. Historical checkpoints and source datasets are unchanged.

## Bounded comparison

A new run started from the same base, source data, seed, learning rate 0.00002,
400 updates and 50/50 update mixture as the previous mixed run. All 763 usable
examples and their shuffled selection remained available; 200 SFT and 200
code-stream updates were performed. The output is
`checkpoints/conversation-turneos-v2-mixed-400step.pt`.

V2 sampled validation loss decreased from 2.1486 to 1.7708 on BigCode and from
4.9940 to 4.8509 on English OASST. Absolute losses across v1 and v2 should not
be directly compared because the supervised target set and serialized context
changed. Each value uses only ten validation examples per source.

Code evaluation passed 9/26 cases, matching the legacy mixed run and remaining
below the base's 13/26. On the same six chat questions, one generation reached
EOS and five reached the 80-token cap, the same counts as the legacy mixture.
Replies remained irrelevant or repetitive; no clear conversation improvement
was found. The termination repair therefore does not explain or solve the full
learning problem in this small, single-seed experiment. Keep the base deployment.

Artifacts: [training report](conversation-turneos-v2-mixed-400step.json),
[chat outputs](chat-turneos-v2-mixed-400step.json),
[code outputs](code-turneos-v2-mixed-400step-samples.json),
[isolated code scores](code-turneos-v2-mixed-400step-scores.json).

Before increasing model size, investigate adequate instruction-training exposure,
data/response diversity and the relation between teacher-forced loss and free
generation, with a separate development protocol and validation-based stopping.
This run visited only 200 of the 763 available SFT examples. Larger models or a
longer run are not assumed to solve these issues automatically.
