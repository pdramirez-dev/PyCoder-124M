"""Likelihood diagnostics for fixed completion choices; no generation or execution."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def score_choice(model, tokenizer, prompt, continuation, device):
    if not prompt or not continuation:
        raise ValueError("Prompt and continuation must be nonempty")
    encoded = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    bos = tokenizer.token_to_id("<|bos|>")
    if bos is None:
        raise ValueError("Tokenizer has no BOS")
    ids = [bos, *encoded.ids]
    if len(ids) - 1 > model.config.context_length:
        raise ValueError("Choice exceeds model context")
    mask = [start >= len(prompt) and end > start for start, end in encoded.offsets]
    if not any(mask):
        raise ValueError("Continuation has no independently scored tokens")
    x = torch.tensor([ids[:-1]], device=device)
    targets = torch.tensor(ids[1:], device=device)
    model.eval()
    logits, _ = model(x)
    losses = F.cross_entropy(logits[0].float(), targets, reduction="none")
    selected = losses[torch.tensor(mask, device=device)]
    return {"tokens": selected.numel(), "nll_sum": selected.sum().item(),
            "nll_mean": selected.mean().item()}
