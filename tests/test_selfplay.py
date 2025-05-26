from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")

from pathlib import Path

from selfrl_chess.mcts import MCTS
from selfrl_chess.network import PolicyValueNet
from selfrl_chess.selfplay import play_one_game, run


def test_play_one_game_returns_aligned_buffers() -> None:
    net = PolicyValueNet(num_filters=4, num_blocks=1).eval()
    mcts = MCTS(net, num_simulations=2, eval_batch=2)
    states, policies, values = play_one_game(mcts, max_plies=4, resign_threshold=None)
    assert len(states) == len(policies) == len(values)
    assert all(s.shape == (17, 8, 8) for s in states)
    assert all(p.shape == (4672,) for p in policies)


def test_run_writes_shard(tmp_path: Path) -> None:
    net = PolicyValueNet(num_filters=4, num_blocks=1).eval()
    shard = run(net, games=1, out_dir=tmp_path, sims=2)
    assert shard.exists()
    assert shard.suffix == ".h5"
