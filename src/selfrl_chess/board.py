"""Board-to-tensor encoding.

The encoder produces a tensor of shape ``(NUM_PLANES, 8, 8)`` (default 17):

    * 0..5     White pieces (P, N, B, R, Q, K).
    * 6..11    Black pieces.
    * 12       Side-to-move plane (1.0 if white to move, else 0.0).
    * 13..16   Castling rights (W kingside, W queenside, B kingside, B queenside).

Setting ``flip_for_black`` mirrors the board so the side to move is always
"at the bottom" -- a trick borrowed from AlphaZero that lets the network share
representations across colours.
"""

from __future__ import annotations

from typing import Final

import chess
import numpy as np

NUM_PLANES: Final = 17
BOARD_SIZE: Final = 8


def board_to_tensor(
    board: chess.Board,
    *,
    flip_for_black: bool = False,
) -> np.ndarray:
    """Encode ``board`` as an ``(NUM_PLANES, 8, 8)`` float32 tensor."""
    tensor = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    perspective_white = (not flip_for_black) or board.turn == chess.WHITE

    for square, piece in board.piece_map().items():
        plane = _piece_to_plane(piece, perspective_white)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        if not perspective_white:
            rank = 7 - rank
            file = 7 - file
        tensor[plane, rank, file] = 1.0

    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    rights = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for i, flag in enumerate(rights, start=13):
        tensor[i, :, :] = 1.0 if flag else 0.0

    return tensor


def _piece_to_plane(piece: chess.Piece, perspective_white: bool) -> int:
    base = piece.piece_type - 1
    is_own = piece.color == chess.WHITE if perspective_white else piece.color == chess.BLACK
    return base if is_own else base + 6
