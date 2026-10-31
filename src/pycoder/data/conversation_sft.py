"""Shared chat formatting with supervised assistant text and end-of-turn EOS."""

import torch

from pycoder.data.conversation import usable_text

IGNORE_INDEX = -100
CHAT_FORMAT = "turn-eos-v2"


def format_turns(messages, add_generation_prompt=False):
    """Return text and supervised character spans using one format in both paths."""
    if not messages:
        raise ValueError("Messages must not be empty")
    text, spans = "", []
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected or not usable_text(message.get("content")):
            raise ValueError("Invalid role, content, or reserved delimiter")
        text += f"<|{expected}|>\n"
        start = len(text)
        text += message["content"]
        if expected == "assistant":
            text += "<|eos|>"
            spans.append((start, len(text)))
        if index < len(messages) - 1 or add_generation_prompt:
            text += "\n"
    if add_generation_prompt:
        if messages[-1]["role"] != "user":
            raise ValueError("Generation must follow a user message")
        text += "<|assistant|>\n"
    elif messages[-1]["role"] != "assistant":
        raise ValueError("Training conversation must end with an assistant reply")
    return text, spans


def encode_conversation(messages, tokenizer, context_length, format_version=CHAT_FORMAT):
    if not messages or len(messages) % 2:
        raise ValueError("A conversation must contain complete user/assistant turns")
    if format_version == CHAT_FORMAT:
        text, spans = format_turns(messages)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        bos, eos = tokenizer.token_to_id("<|bos|>"), tokenizer.token_to_id("<|eos|>")
        if bos is None or eos is None:
            raise ValueError("Tokenizer must contain BOS and EOS")
        ids = [bos, *encoding.ids]
        labels = [IGNORE_INDEX] + [token_id if end > start and any(
            start >= lo and end <= hi for lo, hi in spans) else IGNORE_INDEX
            for token_id, (start, end) in zip(encoding.ids, encoding.offsets, strict=True)]
        if len(ids) - 1 > context_length:
            return None
        if not any(label not in (IGNORE_INDEX, eos) for label in labels):
            raise ValueError("No assistant tokens to train")
        return ids[:-1], labels[1:]
    if format_version != "legacy-v1":
        raise ValueError(f"Unknown chat format: {format_version}")
    text, spans = "", []
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected or not usable_text(message.get("content")):
            raise ValueError("Invalid role, content, or reserved delimiter")
        text += f"<|{expected}|>\n"
        start = len(text)
        text += message["content"]
        if expected == "assistant":
            spans.append((start, len(text)))
        text += "\n"
    encoding = tokenizer.encode(text, add_special_tokens=False)
    bos, eos = tokenizer.token_to_id("<|bos|>"), tokenizer.token_to_id("<|eos|>")
    if bos is None or eos is None:
        raise ValueError("Tokenizer must contain BOS and EOS")
    ids = [bos, *encoding.ids, eos]
    labels = [IGNORE_INDEX]
    for token_id, (start, end) in zip(encoding.ids, encoding.offsets, strict=True):
        labels.append(token_id if end > start and any(start >= lo and end <= hi
                                                       for lo, hi in spans) else IGNORE_INDEX)
    labels.append(eos)
    if len(ids) - 1 > context_length:
        return None
    if not any(label != IGNORE_INDEX for label in labels[1:-1]):
        raise ValueError("No assistant tokens to train")
    return ids[:-1], labels[1:]


def collate_conversations(batch, pad_id):
    if not batch:
        raise ValueError("Batch must not be empty")
    length = max(len(x) for x, _ in batch)
    inputs = torch.full((len(batch), length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)
    for index, (x, y) in enumerate(batch):
        inputs[index, :len(x)] = torch.tensor(x)
        labels[index, :len(y)] = torch.tensor(y)
    return inputs, labels
