import argparse

from pycoder.data.preprocessing import split_files
from pycoder.data.tokenizer import train_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="data/demo")
parser.add_argument("--output", default="data/tokenizer.json")
parser.add_argument("--vocab-size", type=int, default=512)
parser.add_argument("--validation-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
train, _ = split_files(args.source, args.validation_fraction, args.seed)
tok = train_tokenizer([str(path) for path, _ in train], args.output, args.vocab_size)
print(f"Tokenizer saved to {args.output}; vocabulary: {tok.get_vocab_size()}")
