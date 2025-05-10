"""Policy/value network.

Architecture is an AlphaZero-style residual tower with optional
Squeeze-and-Excitation gating:

                  +-----------+
    input ------> |  Stem     |
                  +-----+-----+
                        |
                  +-----v-----+   x N
                  | ResBlock  | -----> ...
                  +-----+-----+
                        |
                  +-----+-----+        +-----+-----+
                  | PolicyHead|        | ValueHead |
                  +-----------+        +-----------+
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with BN, optional SE gating, and a skip path."""

    def __init__(self, channels: int, *, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = _SqueezeExcite(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + x
        return F.relu(out, inplace=True)


class PolicyValueNet(nn.Module):
    """Combined policy and value head over a residual tower."""

    def __init__(
        self,
        input_planes: int = 17,
        num_filters: int = 128,
        num_blocks: int = 10,
        action_size: int = 4672,
        *,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.input_planes = input_planes
        self.action_size = action_size

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(num_filters, use_se=use_se) for _ in range(num_blocks)]
        )

        # Policy head: 1x1 conv to 73 planes, flattened to 4672 logits.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 73, 1, bias=False),
            nn.BatchNorm2d(73),
            nn.ReLU(inplace=True),
        )

        # Value head.
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_policy, value)``."""
        x = self.stem(x)
        x = self.tower(x)

        p = self.policy_conv(x)
        # AlphaZero flattens 73x8x8 in the natural ``(square, plane)`` order
        # used by the encoding module: from_square major, plane minor. Our
        # conv emits ``(plane, rank, file)``, so we permute.
        p = p.permute(0, 2, 3, 1).reshape(p.size(0), -1)
        log_policy = F.log_softmax(p, dim=1)

        v = self.value_conv(x)
        v = v.flatten(start_dim=1)
        value = self.value_fc(v)
        return log_policy, value

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper that returns probabilities (not log-probs)."""
        was_training = self.training
        self.eval()
        try:
            log_p, v = self.forward(x)
            return log_p.exp(), v
        finally:
            if was_training:
                self.train()
