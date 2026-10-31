import json
import subprocess
import sys

import pytest

from pycoder.evaluation.functional import (
    CASES,
    CONTAINER_PROGRAM,
    evaluate_samples,
    extract_function,
    score_sample,
)


def test_extract_only_first_function():
    text = '# add\ndef add(a, b):\n    return a + b\n\ndef add(a, b):\n    return 0\n'
    assert extract_function(text, 'add') == 'def add(a, b):\n    return a + b\n'
    assert extract_function(text, 'missing') is None


def test_harness_scores_trusted_examples():
    payload = json.dumps({'task': 'add', 'code': 'def add(a, b):\n    return a + b',
                          'cases': CASES['add']})
    result = subprocess.run([sys.executable, '-I', '-c', CONTAINER_PROGRAM], input=payload,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 32 + 0b1111
    payload = json.dumps({'task': 'add', 'code': 'def add(a, b):\n    return a - b',
                          'cases': CASES['add']})
    result = subprocess.run([sys.executable, '-I', '-c', CONTAINER_PROGRAM], input=payload,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 32 + 0b0100


def test_container_restrictions_and_scoring(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 32 + 0b1010)

    monkeypatch.setattr('pycoder.evaluation.functional.subprocess.run', run)
    sample = {'task': 'add', 'format': 'comment', 'mode': 'greedy',
              'output': 'def add(a, b):\n    return a + b\n'}
    result = score_sample(sample, 'python:3.12-alpine', 5)
    assert result['case_results'] == [False, True, False, True]
    command = calls[0][0]
    assert '--network=none' in command and '--read-only' in command
    assert '--pull=never' in command and '--cap-drop=ALL' in command
    assert '--user=65534:65534' in command
    assert not any(part.startswith('--volume') or part == '-v' for part in command)


def test_timeout_removes_container(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr('pycoder.evaluation.functional.subprocess.run', run)
    sample = {'task': 'add', 'format': 'x', 'mode': 'greedy', 'output': 'def add(a,b):\n pass'}
    assert score_sample(sample, 'image', 1)['status'] == 'timeout'
    assert calls[1][:3] == ['docker', 'rm', '-f']
    assert calls[1][3].startswith('pycoder-eval-')


def test_missing_docker_fails_closed(monkeypatch, tmp_path):
    samples = tmp_path / 'samples.json'
    samples.write_text(json.dumps({'results': [{'task': 'add'}]}))
    output = tmp_path / 'result.json'

    def fail(*args, **kwargs):
        raise FileNotFoundError('docker')

    monkeypatch.setattr('pycoder.evaluation.functional.subprocess.run', fail)
    with pytest.raises(RuntimeError, match='Docker daemon'):
        evaluate_samples(samples, output, 'python:3.12-alpine', 5)
    assert not output.exists()
