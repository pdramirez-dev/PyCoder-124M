import math

import torch


def make_optimizer(model, config):
    decay, no_decay = [], []
    for param in model.parameters():
        (decay if param.ndim >= 2 else no_decay).append(param)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=config.learning_rate)


def make_scheduler(optimizer, config):
    def factor(step):
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = min(1.0, max(0.0, (step - config.warmup_steps) /
                                     max(1, config.max_steps - config.warmup_steps)))
        floor = config.min_learning_rate / config.learning_rate
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
