import runpy
from pathlib import Path


def test_diverse_filter_checks_syntax_without_executing_code():
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_diverse_pilot.py"
    reject = runpy.run_path(str(script))["rejection_reason"]

    def sample(content, source_id="new"):
        return {"source": "bigcode", "source_id": source_id,
                "messages": [{"role": "assistant", "content": content}]}

    assert reject(sample("```python\ndef bad(:\n```")) == "invalid_python_block"
    assert reject(sample("no code here")) == "missing_python_block"
    assert reject(sample("```python\nraise RuntimeError('must not execute')\n```")) is None
    assert reject(sample("```python\nx = 1\n```", "190969")) == "known_review_issue"
