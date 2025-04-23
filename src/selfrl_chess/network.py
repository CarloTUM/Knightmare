# src/selfrl_chess/network.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Single residual block with two 3x3 conv layers.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = torch.relu(out)
        return out

class PolicyValueNet(nn.Module):
    """
    Combined policy and value network.

    - Input: tensor of shape (B, 17, 8, 8)
    - Policy output: log probabilities over action_size moves
    - Value output: scalar in [-1, 1]
    """
    def __init__(
        self,
        input_planes: int = 17,
        num_filters: int = 128,
        num_blocks: int = 10,
        action_size: int = 4672,
    ):
        super().__init__()
        # Initial convolution
        self.conv = nn.Conv2d(input_planes, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        # Residual tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_blocks)])

        # Policy head
        self.p_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * 8 * 8, action_size)

        # Value head
        self.v_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        # Stem
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        # Residual blocks
        x = self.res_blocks(x)

        # Policy head
        p = self.p_conv(x)
        p = self.p_bn(p)
        p = torch.relu(p)
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)
        p = torch.log_softmax(p, dim=1)

        # Value head
        v = self.v_conv(x)
        v = self.v_bn(v)
        v = torch.relu(v)
        v = v.view(v.size(0), -1)
        v = self.v_fc1(v)
        v = torch.relu(v)
        v = self.v_fc2(v)
        v = torch.tanh(v)

        return p, v
