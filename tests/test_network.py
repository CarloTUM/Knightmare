from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from selfrl_chess.network import PolicyValueNet  # noqa: E402


def test_forward_shapes() -> None:
    net = PolicyValueNet(num_filters=16, num_blocks=2)
    x = torch.randn(3, 17, 8, 8)
    log_p, v = net(x)
    assert log_p.shape == (3, 4672)
    assert v.shape == (3, 1)
    # log-softmax sums to ~1 after exp.
    probs = log_p.exp()
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-4)
    # Value is in (-1, 1).
    assert torch.all(v.abs() <= 1.0)


def test_predict_does_not_change_training_state() -> None:
    net = PolicyValueNet(num_filters=8, num_blocks=1)
    net.train()
    net.predict(torch.randn(1, 17, 8, 8))
    assert net.training


def test_se_can_be_disabled() -> None:
    net = PolicyValueNet(num_filters=8, num_blocks=1, use_se=False)
    log_p, v = net(torch.randn(1, 17, 8, 8))
    assert log_p.shape == (1, 4672)
    assert v.shape == (1, 1)
