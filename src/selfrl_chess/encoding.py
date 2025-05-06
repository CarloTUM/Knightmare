"""AlphaZero-style move encoding.

Each of the 4672 = 64 * 73 actions is identified by:

    index = from_square * 73 + plane

where ``plane`` is one of:

    * 0..55  -- "queen" moves: 8 compass directions x 1..7 squares.
    * 56..63 -- 8 knight moves.
    * 64..72 -- 9 underpromotions: {knight, bishop, rook} x {forward, capture-left, capture-right}.

Queen-promotions reuse the queen-move plane; the chess library implicitly
upgrades the move to a queen promotion when the destination square is the
back rank, which is exactly what AlphaZero does.

This module exposes two helpers:

    move_to_index(move) -> int
    index_to_move(idx, board) -> chess.Move

``index_to_move`` needs the board because we have to know whether the moving
piece is a pawn that just reached the back rank (queen-promotion) or a normal
piece. The mapping is fully reversible for any legal chess move.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import chess

# Standard AlphaZero compass order: (N, NE, E, SE, S, SW, W, NW)
_QUEEN_DIRS: Final = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)

_KNIGHT_DIRS: Final = (
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
)

# Underpromotion: (file_delta, promotion_piece). file_delta in {-1, 0, 1}.
_UNDERPROMO_PIECES: Final = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_UNDERPROMO_FILE_DELTA: Final = (-1, 0, 1)

NUM_PLANES: Final = 73
ACTION_SIZE: Final = 64 * NUM_PLANES  # 4672


def _square_rc(sq: int) -> tuple[int, int]:
    """Return ``(rank, file)`` for ``sq`` (0..63)."""
    return chess.square_rank(sq), chess.square_file(sq)


def _queen_plane(dr: int, df: int) -> int | None:
    """Return the queen-move plane index, or ``None`` if not a queen move."""
    if dr == 0 and df == 0:
        return None
    distance = max(abs(dr), abs(df))
    if distance > 7:
        return None
    if not (dr == 0 or df == 0 or abs(dr) == abs(df)):
        return None
    step_r = 0 if dr == 0 else dr // abs(dr)
    step_f = 0 if df == 0 else df // abs(df)
    try:
        direction = _QUEEN_DIRS.index((step_r, step_f))
    except ValueError:
        return None
    return direction * 7 + (distance - 1)


def _knight_plane(dr: int, df: int) -> int | None:
    try:
        return 56 + _KNIGHT_DIRS.index((dr, df))
    except ValueError:
        return None


def _underpromo_plane(df: int, piece: int) -> int | None:
    try:
        f_idx = _UNDERPROMO_FILE_DELTA.index(df)
        p_idx = _UNDERPROMO_PIECES.index(piece)
    except ValueError:
        return None
    return 64 + p_idx * 3 + f_idx


def move_to_index(move: chess.Move) -> int:
    """Map ``move`` to its AlphaZero action index in ``[0, 4672)``."""
    from_sq = move.from_square
    to_sq = move.to_square
    fr_r, fr_f = _square_rc(from_sq)
    to_r, to_f = _square_rc(to_sq)
    dr, df = to_r - fr_r, to_f - fr_f

    plane: int | None
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Promotions are encoded from the white viewpoint (df sign matches
        # forward direction); for black pawns we mirror df so the encoder is
        # symmetric. The decoder undoes this.
        sign = 1 if to_r > fr_r else -1
        plane = _underpromo_plane(df * sign, move.promotion)
    else:
        plane = _queen_plane(dr, df)
        if plane is None:
            plane = _knight_plane(dr, df)
    if plane is None:
        raise ValueError(f"Move {move.uci()} cannot be encoded")
    return from_sq * NUM_PLANES + plane


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Inverse of :func:`move_to_index`. Needs the board for promotion context."""
    if not 0 <= idx < ACTION_SIZE:
        raise ValueError(f"Action index {idx} out of range")
    from_sq, plane = divmod(idx, NUM_PLANES)
    fr_r, fr_f = _square_rc(from_sq)

    if plane < 56:
        direction, dist = divmod(plane, 7)
        dist += 1
        step_r, step_f = _QUEEN_DIRS[direction]
        to_r, to_f = fr_r + step_r * dist, fr_f + step_f * dist
        promotion: int | None = None
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and to_r in (0, 7):
            promotion = chess.QUEEN
    elif plane < 64:
        dr, df = _KNIGHT_DIRS[plane - 56]
        to_r, to_f = fr_r + dr, fr_f + df
        promotion = None
    else:
        rel = plane - 64
        p_idx, f_idx = divmod(rel, 3)
        df = _UNDERPROMO_FILE_DELTA[f_idx]
        promotion = _UNDERPROMO_PIECES[p_idx]
        sign = 1 if fr_r == 6 else -1  # white pawn on rank 7 -> rank 8
        to_r, to_f = fr_r + sign, fr_f + df * sign

    if not (0 <= to_r < 8 and 0 <= to_f < 8):
        raise ValueError(f"Decoded move {idx} leaves the board")
    to_sq = chess.square(to_f, to_r)
    return chess.Move(from_sq, to_sq, promotion=promotion)


@lru_cache(maxsize=1)
def all_action_indices() -> tuple[int, ...]:
    """All valid action indices (used by tests / sanity checks)."""
    return tuple(range(ACTION_SIZE))
