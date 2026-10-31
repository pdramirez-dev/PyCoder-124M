# 100-example mixed SFT experiment

The trial started from the original 124M checkpoint and used float32 on MPS,
batch size 1, learning rate 2e-5, turn-eos-v2 formatting and 200 alternating
updates: 100 SFT examples and 100 original-code rehearsal sequences. The SFT
data contained 23 reviewed external examples and 77 examples from seven simple
synthetic templates. Validation was evaluated every 50 updates on the first 20
usable examples from each external source. The lowest mean source loss occurred
at step 150, so the saved checkpoint contains 75 SFT updates (58 synthetic,
9 BigCode, 8 OASST) and 75 rehearsal updates, not the last 200-step weights.

| Validation loss | Base | Selected step 150 | Step 200 |
| --- | ---: | ---: | ---: |
| BigCode | 2.0600 | 1.8819 | 1.9012 |
| OASST English | 4.7462 | 4.7803 | 4.8332 |

Lower mean loss did not yield usable conversation. In the six fixed development
chat prompts, the model answered `5` followed by EOS every time; 0/6 responses
met the review criteria. The synthetic data has a clear imbalance: all eleven
word-length examples use five-letter words and have answer `5`. This may have
contributed to the collapse, but the experiment does not establish causation.
The fixed prompts are development checks already used during model iteration.

The same isolated signature-body suite passed 17/26 cases: addition 4/4,
evenness 3/5, factorial 4/4, reversal 4/4, palindrome 2/4 and Fibonacci 0/5.
The base scored 13/26 and the prior 23-example mixed experiment 11/26. This is
a tiny suite and these trials differ in data and budget; the result does not
prove general code improvement. Validation loss and code completion results do
not make this checkpoint a usable chat model.

Artifacts: [training history](conversation-100-mixed-200step.json),
[chat responses](chat-100-mixed-200step.json),
[code samples](code-100-mixed-200step-samples.json),
[isolated scores](code-100-mixed-200step-scores.json). The local weights-only
checkpoint is `checkpoints/conversation-100-mixed-200step.pt`.

Before another SFT trial, diversify the synthetic values and answer lengths,
inspect response-frequency and template balance, and assess generalization on
independent prompts. Preserve this dataset and checkpoint to reproduce the
reported result.

```sh
uv run python scripts/train_conversation_pilot.py checkpoints/latest.pt \
  --data data/processed/conversation-100-en \
  --output checkpoints/conversation-100-mixed-new.pt --device mps \
  --steps 200 --eval-every 50 --learning-rate 2e-5 \
  --rehearsal data/processed/train.bin --rehearsal-context 256
```
