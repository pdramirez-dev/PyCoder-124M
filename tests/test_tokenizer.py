from pycoder.data.preprocessing import prepare, split_files
from pycoder.data.tokenizer import train_tokenizer


def test_tokenizer_and_splits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    samples = ['def add(a, b):\n    return a + b\n', 'def square(x):\n    return x * x\n',
               'def greet():\n    return "hola"\n']
    for i, sample in enumerate(samples + samples[:1]):
        (source / f"{i}.py").write_text(sample)
    train, val = split_files(str(source))
    assert len(train) + len(val) == 3
    tok = train_tokenizer([str(path) for path, _ in train], str(tmp_path / "tok.json"), 300)
    assert tok.decode(tok.encode(samples[0]).ids) == samples[0]
    manifest = prepare(str(source), str(tmp_path / "tok.json"), str(tmp_path / "out"))
    assert set(manifest["train"]).isdisjoint(manifest["val"])
    assert (tmp_path / "out" / "train.bin").stat().st_size > 0
