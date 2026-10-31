# Completion prompt comparison

Final MPS checkpoint, 31 October 2026. Six tasks (addition, evenness,
factorial, string reversal, palindrome and Fibonacci), with 4–5 cases each.
Every prompt produced one completion, capped at 100 new tokens. The first
function definition was evaluated up to the next unindented line; no generated
code was repaired. A task passes only if all of its cases pass.

| Prompt format | Decoding | Cases passed | Tasks passed |
| --- | --- | ---: | ---: |
| Signature, docstring, newline | Greedy | 0/26 | 0/6 |
| Description in comment, then signature | Greedy | 8/26 | 1/6 |
| Signature and docstring, with four spaces after newline | Greedy | 0/26 | 0/6 |
| Same docstring and four-space prefix | Temperature 0.8, top-k 50, seed 42 | 0/26 | 0/6 |
| Signature followed by newline and four spaces | Greedy | 13/26 | 1/6 |

The sum function passed all four cases with the comment and signature-only
formats. With the latter, evenness passed 3/5 cases, factorial 3/4, palindrome
2/4 and Fibonacci 1/5; string reversal passed 0/4. The signature-only prompt
does not describe the function except through its name and arguments.

The docstring-plus-indentation format produced indentation errors for five of
six greedy completions and all six sampled completions. The plain docstring
format consistently produced a docstring-only first function followed by
repeated definitions. The comparison intentionally preserves these exact
outputs; changing or repairing indentation could yield a different result.

The evaluator parsed generated code with a restricted AST and ran allowed code
in separate processes with a three-second limit and limited built-ins. These are
small diagnostic exercises, not HumanEval or MBPP, and the sample size and single
sampling seed do not establish general performance. The corpus was not checked
for benchmark contamination. Full prompts, outputs, evaluated code and individual
case results are in [prompt-formats-results.json](prompt-formats-results.json).

To regenerate these prompts using a trusted local checkpoint and tokenizer:

```sh
uv run python scripts/generate_evaluation_samples.py checkpoints/latest.pt \
  docs/training-runs/mps-2026-10-31/prompt-formats-results.json \
  /tmp/pycoder-new-samples.json --device mps
```

The script generates samples only; it does not run or score them. The destination
must not already exist. Results can vary across devices or software versions even
with the same seed. Execute model output only in a properly isolated environment.

Four local tokenizer round-trip checks, including the docstring prompt with trailing
spaces, tabs and Unicode, preserved their input exactly. This does not prove all
inputs are correct, but the observed extra space after the docstring was generated
by the model rather than introduced by encoding that prompt.
