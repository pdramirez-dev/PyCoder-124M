import json
import os
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

COLUMNS = ["text", "repo_name", "path", "blob_id"]


def sample_parquet(shards, output, max_bytes: int, columns=COLUMNS, text_column="text"):
    """Copy rows from shards, in order, until their UTF-8 text reaches max_bytes."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    writer, rows, total = None, 0, 0
    try:
        for shard in shards:
            file = pq.ParquetFile(shard)
            names = [name for name in columns if name in file.schema_arrow.names]
            if text_column not in names:
                raise ValueError(f"{shard} has no {text_column!r} column")
            for batch in file.iter_batches(batch_size=1024, columns=names):
                batch = batch.filter(pc.is_valid(batch.column(text_column)))
                if not batch.num_rows:
                    continue
                cumulative = np.cumsum(pc.binary_length(batch.column(text_column)).to_numpy())
                keep = min(batch.num_rows,
                           int(np.searchsorted(cumulative, max_bytes - total)) + 1)
                writer = writer or pq.ParquetWriter(tmp, batch.schema)
                writer.write_batch(batch.slice(0, keep))
                rows += keep
                total += int(cumulative[keep - 1])
                if total >= max_bytes:
                    break
            if total >= max_bytes:
                break
    finally:
        if writer:
            writer.close()
    if not rows:
        tmp.unlink(missing_ok=True)
        raise ValueError("No rows with text were found")
    os.replace(tmp, output)
    return rows, total


def download_sample(dataset: str, output, max_bytes: int, pattern="data/train-*.parquet",
                    revision=None, text_column="text"):
    """Download shards lazily until max_bytes of text are sampled; record provenance."""
    info = HfApi().dataset_info(dataset, revision=revision)
    names = sorted(s.rfilename for s in info.siblings if fnmatch(s.rfilename, pattern))
    if not names:
        raise ValueError(f"No files in {dataset} match {pattern!r}")
    used = []

    def shards():
        for name in names:
            used.append(name)
            yield hf_hub_download(dataset, name, repo_type="dataset", revision=info.sha)

    rows, total = sample_parquet(shards(), output, max_bytes, text_column=text_column)
    provenance = {"dataset": dataset, "revision": info.sha, "shards": used, "rows": rows,
                  "text_bytes": total, "text_column": text_column}
    Path(output).with_suffix(".json").write_text(json.dumps(provenance, indent=2),
                                                 encoding="utf-8")
    return provenance
