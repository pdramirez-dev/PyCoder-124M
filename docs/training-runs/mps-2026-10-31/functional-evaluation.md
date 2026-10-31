# Small functional evaluation

Six hand-written Python completion prompts were evaluated against the final MPS
checkpoint. Each prompt supplied the function signature and docstring. One
completion was generated per prompt using greedy decoding (`top_k=1`,
`temperature=1`, maximum 120 new tokens). This is a small diagnostic, not a
standard benchmark or a measure of performance with other prompts or sampling.

For each response, only the first function definition was evaluated: generation
was cut at the first subsequent unindented line. Each first function contained
only the supplied docstring and therefore returned `None`. The model frequently
continued by repeating function definitions, some containing implementations;
these later definitions were not counted as the requested completion.

| Task | Cases passed |
| --- | ---: |
| Add two numbers | 0/3 |
| Check evenness | 0/4 |
| Factorial | 0/3 |
| Reverse a string | 0/3 |
| Check palindrome | 0/3 |
| Fibonacci | 0/3 |
| **Total** | **0/19** |

The exact full decoded outputs are in [functional-samples.json](functional-samples.json).
The evaluated first definitions were inspected as Python ASTs and confirmed to
contain only docstrings before execution. Cases included zero, negative integers,
empty strings, and small positive inputs as appropriate.
