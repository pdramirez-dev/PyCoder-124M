import torch


def find_stop(text, prompt_length, stops):
    """Find the first stop sequence entirely within the generated portion."""
    if any(not stop for stop in stops):
        raise ValueError("Stop sequences cannot be empty")
    positions = [text.find(stop, prompt_length) for stop in stops]
    return min((position for position in positions if position >= 0), default=None)


@torch.no_grad()
def generate(model, tokens, max_new_tokens=100, temperature=1.0, top_k=None, eos_id=None,
             vocab_limit=None, stop_callback=None):
    if temperature <= 0 or max_new_tokens < 0 or (top_k is not None and top_k <= 0):
        raise ValueError("Invalid generation parameters")
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(tokens[:, -model.config.context_length:])
            logits = logits[:, -1, :] / temperature
            if vocab_limit is not None:
                if not 0 < vocab_limit <= logits.size(-1):
                    raise ValueError("Invalid vocabulary limit")
                logits = logits[:, :vocab_limit]
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
            if stop_callback is not None and stop_callback(tokens):
                break
    finally:
        model.train(was_training)
    return tokens
