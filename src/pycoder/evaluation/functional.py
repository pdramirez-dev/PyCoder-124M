"""Functional scoring of saved completions using isolated Docker containers."""

import json
import re
import subprocess
import uuid
from pathlib import Path

CASES = {
    "add": [([2, 3], 5), ([-2, 5], 3), ([0, 0], 0), ([-4, -6], -10)],
    "is_even": [([0], True), ([2], True), ([3], False), ([-2], True), ([-3], False)],
    "factorial": [([0], 1), ([1], 1), ([5], 120), ([7], 5040)],
    "reverse_string": [([""], ""), (["abc"], "cba"), (["a"], "a"),
                       (["hello world"], "dlrow olleh")],
    "is_palindrome": [([""], True), (["abba"], True), (["abc"], False), (["a"], True)],
    "fibonacci": [([0], 0), ([1], 1), ([2], 1), ([6], 8), ([10], 55)],
}

# The only data leaving the container is the exit status. 32..63 encode the
# pass bitmask for the at most five cases in the fixed suite.
CONTAINER_PROGRAM = r"""
import json, sys
p = json.load(sys.stdin)
try:
    scope = {}
    exec(compile(p['code'], '<completion>', 'exec'), scope)
    fn = scope[p['task']]
    mask = 0
    for i, (args, expected) in enumerate(p['cases']):
        try:
            actual = fn(*args)
            if type(actual) is type(expected) and actual == expected:
                mask |= 1 << i
        except BaseException:
            pass
except BaseException:
    sys.exit(10)
sys.exit(32 + mask)
"""


def extract_function(output, name):
    """Extract the first target definition through the next top-level line."""
    match = re.search(r"^def " + re.escape(name) + r"\(", output, re.MULTILINE)
    if not match:
        return None
    return re.split(r"\n(?=\S)", output[match.start():], maxsplit=1)[0]


def docker_command(image, name):
    return [
        "docker", "run", "--rm", "--pull=never", "--name", name, "-i",
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--pids-limit=32", "--memory=256m", "--cpus=1", "--user=65534:65534",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m", image, "python", "-I", "-c",
        CONTAINER_PROGRAM,
    ]


def check_docker(image):
    """Fail before evaluation if the daemon or the requested local image is unavailable."""
    try:
        subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        subprocess.run(["docker", "image", "inspect", image], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Docker daemon or local evaluation image is unavailable") from error


def score_sample(sample, image, timeout):
    task = sample["task"]
    if task not in CASES:
        raise ValueError(f"Unknown task: {task}")
    code = extract_function(sample["output"], task)
    result = {key: sample[key] for key in ("task", "format", "mode")}
    result["total"] = len(CASES[task])
    if code is None:
        return result | {"status": "missing_function", "case_results": []}
    if len(code.encode("utf-8")) > 100_000:
        return result | {"status": "too_large", "case_results": []}
    container = "pycoder-eval-" + uuid.uuid4().hex
    payload = json.dumps({"task": task, "code": code, "cases": CASES[task]})
    try:
        process = subprocess.run(docker_command(image, container), input=payload, text=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15, check=False)
        except (subprocess.TimeoutExpired, OSError) as error:
            raise RuntimeError(f"Could not clean up timed-out container {container}") from error
        return result | {"status": "timeout", "case_results": []}
    if process.returncode == 10:
        return result | {"status": "invalid_or_error", "case_results": []}
    upper = 32 + (1 << len(CASES[task]))
    if not 32 <= process.returncode < upper:
        raise RuntimeError(f"Docker failed or container returned unexpected status {process.returncode}")
    mask = process.returncode - 32
    passed = [bool(mask & (1 << index)) for index in range(len(CASES[task]))]
    return result | {"status": "evaluated", "case_results": passed}


def evaluate_samples(input_path, output_path, image, timeout):
    """Score a sample file; never replace an existing result file."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    records = json.loads(Path(input_path).read_text(encoding="utf-8"))["results"]
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty results list")
    check_docker(image)
    scored = []
    for sample in records:
        result = score_sample(sample, image, timeout)
        scored.append(result)
        print(f"{result['task']}: {result['status']} "
              f"{sum(result['case_results'])}/{result['total']}", flush=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump({"suite": "pycoder-small-v1", "image": image, "timeout_seconds": timeout,
                   "results": scored}, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return scored
