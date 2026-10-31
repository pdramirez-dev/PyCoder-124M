# 10k synthetic pilot: 1,000 mixed updates

The pilot starts from the original 124M checkpoint using float32 MPS, batch 1,
learning rate 2e-5, turn-eos-v2 and validation every 100 steps. It alternates
500 SFT updates with 500 original-code rehearsal updates (context 256). With
the deterministic shuffle, SFT saw 497 synthetic examples, two BigCode and
one OASST example: 500 of 10,000 training records, not a complete pass.

The selected checkpoint was step 0 because no measured point improved the
mean of the first 20 usable validation examples per source. In the repeat run,
BigCode loss changed from 2.0600 at step 0 to 2.0864 at step 1000; OASST loss
changed from 4.7462 to 4.9005. These are teacher-forced losses and cover only a
small sample. The first run saved only the base weights selected by validation.
The experiment was repeated with an explicit separate final-weight output to
allow assessment of what the training updates actually produced.

The final-step checkpoint produced no satisfactory answers on the six fixed
development chat prompts and scored 0/11 exact answers on the new-value
synthetic diagnostic. That diagnostic uses familiar task formats and is not a
final benchmark. The isolated code suite passed 4/26 cases (addition 4/4;
all other tasks 0), versus 13/26 for the base in this small suite. These
results do not prove why the behavior deteriorated, but they do not support
continuing this data mixture unchanged.

Artifacts: [training history](conversation-10k-mixed-1000step.json),
[chat outputs](chat-10k-mixed-1000step-final.json),
[new-value outputs](generalization-10k-mixed-1000step-final.json),
[code samples](code-10k-mixed-1000step-final-samples.json), and
[isolated scores](code-10k-mixed-1000step-final-scores.json).
The local `checkpoints/conversation-10k-mixed-1000step-final.pt` holds the
final weights; `checkpoints/conversation-10k-mixed-1000step-selected-replay.pt`
holds the validation-selected base weights. These are weights-only outputs and
do not permit exact optimizer resume. The report's `saved_weights_step` marks
the final checkpoint; `best_validation_step` describes selection, not its weights.

The mixture is 99.77% synthetic in the SFT dataset and uses few templates.
Future work should improve data quality and diversity and explicitly control
sampling by source before a longer run. Preserve the held-out data and use
independent prompts for assessment.
