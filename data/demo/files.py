from pathlib import Path


def python_files(directory):
    return sorted(path for path in Path(directory).rglob("*.py") if path.is_file())


def line_count(path):
    with open(path, encoding="utf-8") as stream:
        return sum(1 for _ in stream)
