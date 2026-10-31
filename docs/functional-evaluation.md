# Functional evaluation

This small, fixed diagnostic suite compares generated Python completions against
26 cases across six tasks. It is not a standard benchmark; the training corpus
has not been checked for overlap with the prompts. Keep the checkpoint,
tokenizer, sample JSON, result JSON and environment versions for comparisons.

## Generate completions

From the repository root, using the completed MPS checkpoint as an example:

```sh
uv run python scripts/generate_evaluation_samples.py checkpoints/latest.pt \
  docs/training-runs/mps-2026-10-31/prompt-formats-results.json \
  /tmp/pycoder-samples.json --device mps
```

The output must not already exist. The input file supplies six tasks and five
prompt/decoding combinations; sampling uses seed 42. Results may still differ
across PyTorch versions, devices and backends. This stage never executes model
output. Checkpoints must come from a trusted source.

## Score completions

Install and start Docker Desktop (or a Docker-compatible daemon), and obtain a
trusted local `python:3.12-alpine` image once:

```sh
docker pull python:3.12-alpine
```

Then run the scorer. It will not pull images automatically, and it will not
execute generated Python on the host if Docker or the local image is missing.

```sh
uv run python scripts/evaluate_functional.py /tmp/pycoder-samples.json \
  /tmp/pycoder-scores.json
```

To score the samples already collected during the first training run instead,
use `docs/training-runs/mps-2026-10-31/prompt-formats-results.json` as the input.
Use `--image` to select a trusted local image, preferably one identified by its
digest for repeatability. The image must contain a `python` executable. Use
`--timeout` to adjust the per-completion wall-clock limit, which includes
container startup. Output paths are never overwritten.

Every completion runs in a fresh container with no network or host mounts, a
read-only root filesystem, an unprivileged user, restricted capabilities,
process, memory and CPU limits, and a bounded temporary filesystem. Docker is an
isolation layer, not a perfect security boundary: use an up-to-date runtime and
trusted image, and use a disposable VM for stronger isolation. Generated code
must not be run directly on the host. The scoring protocol is intended for
accidental errors, not for defending the validity of results against a model
intentionally trying to forge scores.

The evaluator extracts the first requested function, ending at the next
unindented line; it does not repair syntax or indentation. Each row records
case-level booleans, `invalid_or_error`, `missing_function`, `too_large` or
`timeout`. A task succeeds only when all of its cases pass. Results are saved
only after the full run; Docker infrastructure errors stop the process. The
timeout cleanup removes the named container. The suite and scoring logic are in
`src/pycoder/evaluation/functional.py`.

The first isolated run of the saved MPS samples is recorded in
[`training-runs/mps-2026-10-31/functional-docker-results.json`](training-runs/mps-2026-10-31/functional-docker-results.json).
It used `python:3.12-alpine` with manifest digest
`sha256:e6dae589c1c05976188fd3bd019afd091445fc533ed2ea29001beddc7368a9c8`.
