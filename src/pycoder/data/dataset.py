import hashlib

import numpy as np
import torch


class TokenStream:
    """Deterministic cycling stream; position is measured in token offsets."""
    def __init__(self, path: str, context_length: int):
        with open(path, "rb") as file:
            self.fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")
        self.context_length = context_length
        if len(self.tokens) <= context_length:
            raise ValueError(f"{path} needs more than {context_length} tokens")
        self.position = 0

    def batch(self, batch_size: int, device: torch.device):
        n = len(self.tokens) - self.context_length
        offsets = [(self.position + i * self.context_length) % n for i in range(batch_size)]
        self.position = (self.position + batch_size * self.context_length) % n
        samples = np.stack([self.tokens[i:i + self.context_length + 1] for i in offsets]).astype(np.int64)
        return (torch.from_numpy(samples[:, :-1].copy()).to(device),
                torch.from_numpy(samples[:, 1:].copy()).to(device))
