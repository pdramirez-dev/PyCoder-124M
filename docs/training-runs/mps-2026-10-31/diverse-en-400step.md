# Broader English pilot with code rehearsal

Starting from the original 124M checkpoint, a fresh trial used the broader English
pilot after structural filtering and explicit exclusions for known review issues.
The dataset contains 785 conversations: 394 BigCode and 391 OASST English. It
includes 467 two-message, 292 four-message and 26 six-message conversations.
Thirteen examples with identified issues and one example with invalid Python
syntax were excluded. Syntax checks do not establish functional correctness,
and the remaining corpus has not been fully reviewed or decontaminated.

Twenty-two training examples exceed context length and were skipped, leaving
763 usable examples. The held-out splits were preserved. All original data,
the base checkpoint, tokenizer and Ollama model remain unchanged.

The trial ran 400 float32 updates on MPS, batch size 1, learning rate 0.00002.
Updates alternated between SFT and the original Python training stream with
context 256: 200 updates of each type. This is equal update frequency, not equal
token or gradient weighting. The 200 SFT updates visited 200 different examples
in deterministic shuffled order. Context size for SFT remained up to 1024.
Several factors changed from the prior trial, so improvement cannot be attributed
to rehearsal or diversity alone without controlled ablations.

## Comparison

| Diagnostic | Base | 11 examples / 100 updates | Broader mixed trial |
| --- | ---: | ---: | ---: |
| BigCode validation loss, same 10 examples | 2.1461 | 2.1673 | 1.7658 |
| OASST English validation loss, same 10 examples | 4.9733 | 5.5272 | 4.8610 |
| Signature-only Python cases passed | 13/26 | 0/26 | 9/26 |
| Functions passing every case | 1/6 | 0/6 | 1/6 |

Isolated code scoring: addition 4/4, evenness 3/5, factorial 1/4, reversal 0/4,
palindrome 0/4, Fibonacci 1/5. The code evaluator uses the same prompts, greedy
generation, 100-token cap and first-definition extraction as the base comparison.

The broader trial improves the sampled teacher-forced validation loss and retains
more code performance than the tiny overfitting trial, but still falls below the
base code score. The six fixed chat prompts still produced irrelevant or
repetitive replies; five reached the 80-token cap and one reached EOS. None was
judged satisfactory on manual review. These development prompts have already
influenced experiments and are not an untouched final benchmark. The corpus
test split has not been used for model selection.

The checkpoint `checkpoints/conversation-diverse-en-400step.pt` is an analysis
artifact, not a replacement for the base. No GGUF or Ollama deployment was made
from it. These results do not justify scaling this recipe directly to 1B.

Artifacts: [training metrics and hashes](conversation-diverse-en-400step.json),
[chat outputs](chat-diverse-en-400step.json),
[code outputs](code-diverse-en-400step-samples.json),
[isolated scores](code-diverse-en-400step-scores.json).

## Reproduction

```sh
uv run python scripts/prepare_diverse_pilot.py
uv run python scripts/train_conversation_pilot.py checkpoints/latest.pt \
  --data data/processed/conversation-diverse-en \
  --output checkpoints/conversation-diverse-en-new.pt --device mps \
  --steps 400 --learning-rate 0.00002 \
  --chat-format legacy-v1 \
  --rehearsal data/processed/train.bin --rehearsal-context 256
```

Use a new dataset directory or checkpoint path if outputs already exist. This
pilot writes a weights-only checkpoint and does not support exact resume.
Next investigations should control data size, update budget and rehearsal
separately, increase review coverage, and examine general English pretraining
competence before using instruction tuning alone to obtain conversation ability.
