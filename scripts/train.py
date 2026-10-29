import argparse

from tokenizers import Tokenizer

from pycoder.config import ModelConfig, TrainConfig, load_config
from pycoder.training.trainer import train

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="configs/model_tiny.yaml")
parser.add_argument("--config", default="configs/train.yaml")
parser.add_argument("--data", default="data/processed")
parser.add_argument("--tokenizer", default="data/tokenizer.json")
parser.add_argument("--checkpoints", default="checkpoints")
parser.add_argument("--logs", default="logs")
parser.add_argument("--resume")
parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
args = parser.parse_args()
model_config = load_config(args.model, ModelConfig)
if Tokenizer.from_file(args.tokenizer).get_vocab_size() > model_config.vocab_size:
    raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
train(model_config, load_config(args.config, TrainConfig), args.data, args.checkpoints, args.logs,
      args.resume, args.device)
