"""Exponential Moving Average of model parameters.

This is a standard regularizer for AlphaZero-style training: the EMA shadow
network usually outperforms the raw network by ~50 Elo.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain a shadow copy of ``model`` updated as
    ``shadow = decay * shadow + (1 - decay) * model``.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self._params(self.module), self._params(model), strict=True):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        # Also copy buffers (BN running stats) verbatim -- they are not learnable.
        for ema_b, b in zip(self.module.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    @staticmethod
    def _params(module: nn.Module) -> Iterable[torch.Tensor]:
        return (p for p in module.parameters() if p.dtype.is_floating_point)
