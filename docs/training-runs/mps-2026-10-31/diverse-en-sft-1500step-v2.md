# Diverse English SFT: 1,500 updates

This trial starts from the original `checkpoints/latest.pt` and uses the same
763-example structurally filtered English training split as the 400-step v2
trial. It applies 1,500 SFT updates at learning rate `2e-5`, batch size 1,
context up to 1,024, and the `turn-eos-v2` format. The data order is shuffled
and reshuffled deterministically. Each training update uses one conversation;
the run sees roughly two passes over the available examples.

The training script now evaluates every 100 updates on the first 20 usable
validation conversations per source and saves the weights with the lowest
mean of the two source losses. It selected step 1,300. Validation comparisons
below use the same 20 rows before and after this run:

| Source | Base | Selected step 1,300 |
| --- | ---: | ---: |
| BigCode | 2.0600 | 1.5576 |
| OASST English | 4.7462 | 4.4863 |

Those losses are teacher-forced token prediction metrics. They do not establish
that free-form answers are useful. The fixed six-prompt chat review remained
poor: only the greeting returned a short relevant answer; the other five
outputs were irrelevant or repetitive and hit the 80-token limit.

The same six signature-body code prompts scored 4/26 cases (addition 4/4;
the other five tasks generated invalid or unextractable code). The earlier
400-update v2 mixed run scored 9/26. That earlier run used 200 SFT and 200 code
rehearsal updates, so this is not a controlled comparison; it does show that
more SFT updates by themselves have not produced a better usable model here.

This checkpoint is an experiment artifact, not a replacement for the base and
not a candidate for the 1B scale-up. The actionable result is to improve and
review the instruction data and response behavior before increasing model
size or extending this SFT recipe. The fixed development prompts have already
guided iteration and are not an untouched final benchmark.

Artifacts: [checkpoint metadata and loss history](conversation-diverse-en-sft-1500step-v2.json),
[chat generations](chat-diverse-en-sft-1500step-v2.json),
[code samples](code-diverse-en-sft-1500step-v2-samples.json),
[isolated code scores](code-diverse-en-sft-1500step-v2-scores.json).

## Reproduction

```sh
uv run python scripts/train_conversation_pilot.py checkpoints/latest.pt \
  --data data/processed/conversation-diverse-en \
  --tokenizer data/tokenizer.json \
  --output checkpoints/conversation-diverse-en-sft-new.pt --device mps \
  --steps 1500 --eval-every 100 --learning-rate 0.00002
```
