import argparse

from pycoder.data.download import download_sample

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Avelina/python-edu-cleaned")
parser.add_argument("--output", default="data/raw/python-edu.parquet")
parser.add_argument("--max-mb", type=float, default=400, help="UTF-8 text to sample, in MB")
parser.add_argument("--pattern", default="data/train-*.parquet")
parser.add_argument("--revision")
parser.add_argument("--text-column", default="text")
args = parser.parse_args()
info = download_sample(args.dataset, args.output, int(args.max_mb * 1e6), args.pattern,
                       args.revision, args.text_column)
print(f"Saved {info['rows']:,} files ({info['text_bytes'] / 1e6:.1f} MB) from "
      f"{len(info['shards'])} shard(s) at revision {info['revision'][:12]} to {args.output}")
