from __future__ import annotations

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from selfrl_chess.mcts import MCTS, _terminal_value
from selfrl_chess.network import PolicyValueNet


def _tiny_net() -> PolicyValueNet:
    return PolicyValueNet(num_filters=8, num_blocks=1)


def test_terminal_value_detects_checkmate() -> None:
    # Fool's mate: 1. f3 e5 2. g4 Qh4#
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)
    assert board.is_checkmate()
    # Side-to-move (white) lost.
    assert _terminal_value(board) == -1.0


def test_policy_returns_valid_distribution_on_starting_position() -> None:
    net = _tiny_net().eval()
    mcts = MCTS(net, num_simulations=8, eval_batch=4)
    pi = mcts.policy(chess.Board(), temperature=1.0, add_noise=False)
    assert pi.shape == (4672,)
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)
    assert (pi >= 0).all()


def test_policy_concentrates_on_legal_moves() -> None:
    net = _tiny_net().eval()
    mcts = MCTS(net, num_simulations=4, eval_batch=2)
    board = chess.Board()
    pi = mcts.policy(board, temperature=1.0, add_noise=False)
    # Mass should sit on legal moves only.
    from selfrl_chess.encoding import move_to_index

    legal_indices = {move_to_index(m) for m in board.legal_moves}
    illegal_mass = sum(p for i, p in enumerate(pi) if i not in legal_indices)
    assert illegal_mass < 1e-6


def test_advance_reuses_subtree() -> None:
    net = _tiny_net().eval()
    mcts = MCTS(net, num_simulations=4, eval_batch=2)
    board = chess.Board()
    mcts.policy(board, temperature=1.0, add_noise=False)
    move = next(iter(mcts.root.children))
    mcts.advance(move)
    assert mcts.root is not None
    assert mcts.root.parent is None
