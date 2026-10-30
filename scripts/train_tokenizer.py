import argparse

from pycoder.config import FilterConfig, load_config
from pycoder.data.preprocessing import split_files
from pycoder.data.tokenizer import train_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="data/demo")
parser.add_argument("--output", default="data/tokenizer.json")
parser.add_argument("--vocab-size", type=int, default=512)
parser.add_argument("--validation-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--text-column", default="text")
parser.add_argument("--filters", default="configs/filters.yaml")
args = parser.parse_args()
train, _, rejected = split_files(args.source, args.validation_fraction, args.seed,
                                 args.text_column, load_config(args.filters, FilterConfig))
tok = train_tokenizer([content for _, content in train], args.output, args.vocab_size)
print(f"Rejected: {rejected or 'none'}")
print(f"Tokenizer saved to {args.output}; vocabulary: {tok.get_vocab_size()}")
