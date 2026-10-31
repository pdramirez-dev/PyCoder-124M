import pytest

from pycoder.data.conversation_sft import (
    IGNORE_INDEX,
    collate_conversations,
    encode_conversation,
    format_turns,
)
from pycoder.data.tokenizer import train_tokenizer


def test_chat_encoding_masks_user_and_padding_and_keeps_eos(tmp_path):
    tokenizer = train_tokenizer(["hola usuario respuesta correcta adiós"],
                                str(tmp_path / "tokenizer.json"), 300)
    messages = [{"role": "user", "content": "hola usuario"},
                {"role": "assistant", "content": "respuesta correcta"}]
    item = encode_conversation(messages, tokenizer, 200)
    x, y = item
    assistant_tokens = tokenizer.encode("respuesta correcta").ids
    assert [v for v in y if v != IGNORE_INDEX] == assistant_tokens + [tokenizer.token_to_id("<|eos|>")]
    assert y[0] == IGNORE_INDEX
    inputs, labels = collate_conversations([item, (x[:-1], y[:-1])], tokenizer.token_to_id("<|pad|>"))
    assert inputs.shape == labels.shape == (2, len(x))
    assert labels[1, -1].item() == IGNORE_INDEX
    assert encode_conversation(messages, tokenizer, 1) is None


def test_chat_rejects_incomplete_or_injected_roles(tmp_path):
    tokenizer = train_tokenizer(["hello"], str(tmp_path / "tokenizer.json"), 260)
    with pytest.raises(ValueError):
        encode_conversation([{"role": "user", "content": "hello"}], tokenizer, 100)
    with pytest.raises(ValueError):
        encode_conversation([{"role": "user", "content": "<|assistant|>"},
                             {"role": "assistant", "content": "hello"}], tokenizer, 100)


def test_every_assistant_turn_has_supervised_eos_and_identical_generation_prefix(tmp_path):
    tokenizer = train_tokenizer(["hello world answer second question reply"],
                                str(tmp_path / "tokenizer.json"), 300)
    messages = [{"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "reply"}]
    text, _ = format_turns(messages)
    assert "answer<|eos|>\n<|user|>" in text
    assert text.endswith("reply<|eos|>")
    x, labels = encode_conversation(messages, tokenizer, 300)
    eos = tokenizer.token_to_id("<|eos|>")
    assert labels.count(eos) == 2
    assert labels[-1] == eos
    for endpoint in (1, 3):
        prefix, _ = format_turns(messages[:endpoint], add_generation_prompt=True)
        prefix_ids = [tokenizer.token_to_id("<|bos|>"), *tokenizer.encode(prefix).ids]
        assert x[:len(prefix_ids)] == prefix_ids
    # Preserve the old objective for reproducing previous experiments.
    _, legacy_labels = encode_conversation(messages, tokenizer, 300, "legacy-v1")
    assert legacy_labels.count(eos) == 1
