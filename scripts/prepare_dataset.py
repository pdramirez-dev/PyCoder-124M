import argparse

from pycoder.data.preprocessing import prepare

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="data/demo")
parser.add_argument("--tokenizer", default="data/tokenizer.json")
parser.add_argument("--output", default="data/processed")
parser.add_argument("--validation-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
manifest = prepare(args.source, args.tokenizer, args.output, args.validation_fraction, args.seed)
print(f"Prepared {len(manifest['train'])} training and {len(manifest['val'])} validation files")
