from __future__ import annotations

import chess
import numpy as np

from selfrl_chess.board import BOARD_SIZE, NUM_PLANES, board_to_tensor


def test_starting_position_shape_and_planes() -> None:
    t = board_to_tensor(chess.Board())
    assert t.shape == (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert t.dtype == np.float32

    # 16 white pieces and 16 black pieces total.
    white = t[:6].sum()
    black = t[6:12].sum()
    assert white == 16
    assert black == 16

    # Side-to-move plane.
    assert (t[12] == 1.0).all()
    # All castling rights are set in the starting position.
    assert (t[13:17].sum(axis=(1, 2)) == np.array([64, 64, 64, 64], dtype=np.float32)).all()


def test_flip_for_black_swaps_perspective() -> None:
    board = chess.Board()
    board.push_san("e4")  # Black to move.
    flipped = board_to_tensor(board, flip_for_black=True)
    assert flipped.shape == (NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    # In the flipped representation black pieces should appear on planes 0..5
    # because they are now "own".
    assert flipped[0:6].sum() == 16  # own pieces (= black) are now in own planes
    assert flipped[6:12].sum() == 16
