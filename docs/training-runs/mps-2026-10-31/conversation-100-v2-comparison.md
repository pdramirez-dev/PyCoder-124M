# Comparison of 100-example SFT pilots

The v2 trial starts from the same base checkpoint, tokenizer and reviewed 23
external conversations as v1, replacing the other 77 synthetic examples. Both
runs use seed 42, float32 MPS, batch 1, learning rate 2e-5, 200 alternating
updates, code rehearsal context 256 and validation every 50 steps. Both selected
step 150: 58 synthetic, 9 BigCode, 8 OASST and 75 rehearsal updates. Different
synthetic texts imply different token counts; this is not a token-matched trial.

| Diagnostic | v1 | v2 |
| --- | ---: | ---: |
| BigCode validation loss at selected checkpoint | 1.8819 | 1.8759 |
| OASST validation loss at selected checkpoint | 4.7803 | 4.7570 |
| Development chat prompts judged satisfactory | 0/6 | 0/6 |
| Signature-body code cases passed | 17/26 | 16/26 |
| New synthetic-task prompts, exact answer | 0/11 | 0/11 |

At the selected checkpoint the shared base losses were 2.0600 and 4.7462 on
the first 20 usable validation conversations per source. v1 answered `5` on
all six development chat prompts; v2 produced several different replies but
none met the review criteria. Its answers included irrelevant code, incorrect
short responses and repetition. More varied output alone is not an improvement
in instruction following.

The 11 new prompts use new strings, input combinations and expected results,
and have no normalized exact prompt overlap with either training set. They use
the same task families and mostly the same phrasing as v2 training, so they
measure limited in-family transfer, not broad conversational generalization.
Two task families appear in v2 but not v1, limiting direct attribution. The suite
was authored after the datasets, is tiny and is now a development diagnostic;
pretraining or paraphrase overlap has not been ruled out. Exact-answer scores
compare stripped replies with the recorded expected string. None matched for
either checkpoint. Preserve an independently constructed final benchmark.

The code suite uses the same greedy prompts and Docker isolation as the earlier
report. V2 scored addition 4/4, evenness 2/5, factorial 4/4, reversal 4/4,
palindrome 2/4 and Fibonacci 0/5. Differences of one case on this small suite
do not establish a meaningful regression or improvement.

Artifacts: [training history](conversation-100-v2-mixed-200step.json),
[chat results](chat-100-v2-mixed-200step.json),
[v1 new-value results](generalization-100-v1.json),
[v2 new-value results](generalization-100-v2.json),
[code samples](code-100-v2-mixed-200step-samples.json),
[isolated code scores](code-100-v2-mixed-200step-scores.json), and
[diagnostic prompts](../../../configs/synthetic_generalization_eval_en.json).
The weights-only checkpoint is local at `checkpoints/conversation-100-v2-mixed-200step.pt`.

This experiment does not justify scaling to 1B or extending the small synthetic
recipe. Broader, reviewed instruction data and an independent held-out suite are
needed before another substantive training experiment.
