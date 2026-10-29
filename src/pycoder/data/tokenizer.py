from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]


def train_tokenizer(files: list[str], output: str, vocab_size: int) -> Tokenizer:
    if vocab_size < 260:
        raise ValueError("Byte-level BPE requires at least 260 tokens")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train(files, trainer)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output)
    return tokenizer
