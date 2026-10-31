# Source-weighted 10k SFT pilot

This trial used the original 124M base, the same 10k training split, seed,
1,000-update budget, 50/50 SFT/rehearsal update schedule and validation as the
[previous 10k experiment](conversation-10k-mixed-1000step.md). During SFT it
sampled sources with relative weights 90 synthetic, 5 BigCode and 5 OASST, with
maximums of 22 and 24 updates for the 11 and 12 reviewed examples respectively.
Each source is shuffled independently without replacement until exhausted,
then reshuffled. Once a source reaches its update cap it is removed from the
sampling pool. The cap allows at most two visits per reviewed example.

Across all 1,000 steps, the counts were 454 synthetic, 22 BigCode, 24 OASST
and 500 code rehearsal. The best mean validation loss occurred at step 300,
after 128 synthetic, 11 BigCode, 11 OASST and 150 rehearsal updates. The
selected checkpoint's BigCode and OASST losses were 1.8759 and 4.7678; the
base measured 2.0600 and 4.7462 on the first 20 examples per source. The older
uniform-sampling experiment selected step 0. Source sampling order and selected
steps differ, and variable sequence lengths mean this is not token-matched.

The selected checkpoint produced 0/6 satisfactory development chat answers,
1/11 exact answers on the new-value diagnostic (a yes/no parity question), and
10/26 code cases in the isolated signature-body suite: addition 4/4, evenness
2/5, factorial 2/4, reversal 0/4, palindrome 2/4 and Fibonacci 0/5. These small
development suites cannot establish generalization; one parity answer may be
correct by chance. The previous 10k run's *final* weights scored 4/26, but its
validation-selected weights were the base. Comparing different selected steps
does not isolate the source-weighting effect. The base scored 13/26.

The mixture improves exposure to reviewed examples but still does not produce
reliable chat and remains mostly synthetic. A wider reviewed instruction corpus
and more independent evaluation are needed before a longer or larger run.
Original data source-license and provenance obligations remain to be reviewed.

Artifacts: [training history](conversation-10k-balanced-1000step.json),
[chat outputs](chat-10k-balanced-selected.json),
[new-value outputs](generalization-10k-balanced-selected.json),
[code outputs](code-10k-balanced-selected-samples.json) and
[isolated code scores](code-10k-balanced-selected-scores.json). Weights-only
local checkpoints are `checkpoints/conversation-10k-balanced-1000step-selected.pt`
and `checkpoints/conversation-10k-balanced-1000step-final.pt`; neither includes
optimizer state for exact resume.

```sh
uv run python scripts/train_conversation_pilot.py checkpoints/latest.pt \
  --data data/processed/conversation-10k-en \
  --output checkpoints/conversation-10k-balanced-new.pt \
  --final-output checkpoints/conversation-10k-balanced-final-new.pt \
  --source-mixture configs/conversation_10k_source_mixture.json \
  --device mps --steps 1000 --eval-every 100 --learning-rate 2e-5 \
  --rehearsal data/processed/train.bin --rehearsal-context 256
```
