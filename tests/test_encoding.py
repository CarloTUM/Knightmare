"""Round-trip tests for the AlphaZero move encoding.

The encoding has to be reversible for *every* legal chess move that can
appear in a game, including castling, en passant, and all four promotion
piece types. We assert that exhaustively across a battery of positions.
"""

from __future__ import annotations

import random

import chess
import chess.pgn
import pytest

from selfrl_chess.encoding import ACTION_SIZE, NUM_PLANES, index_to_move, move_to_index


def test_action_size_is_4672() -> None:
    assert ACTION_SIZE == 4672
    assert NUM_PLANES == 73
    assert ACTION_SIZE == 64 * NUM_PLANES


def test_indices_are_unique_per_position() -> None:
    """Two different legal moves never collide on the same action index."""
    board = chess.Board()
    indices = {move_to_index(m) for m in board.legal_moves}
    assert len(indices) == len(list(board.legal_moves))


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        # Mid-game position with rich move set.
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        # Pre-castling position.
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        # En-passant available.
        "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2",
        # White pawn poised to promote.
        "8/P7/8/8/8/8/k7/4K3 w - - 0 1",
        # Black pawn poised to promote.
        "4k3/K7/8/8/8/8/p7/8 b - - 0 1",
    ],
)
def test_round_trip_for_all_legal_moves(fen: str) -> None:
    board = chess.Board(fen)
    for move in board.legal_moves:
        idx = move_to_index(move)
        assert 0 <= idx < ACTION_SIZE
        decoded = index_to_move(idx, board)
        assert decoded == move, f"round-trip failed for {move.uci()} ({fen})"


def test_round_trip_through_random_game() -> None:
    """Walk a random game; every move must round-trip on the live board."""
    rng = random.Random(0)
    board = chess.Board()
    for _ in range(80):
        if board.is_game_over(claim_draw=True):
            break
        legal = list(board.legal_moves)
        move = rng.choice(legal)
        idx = move_to_index(move)
        assert index_to_move(idx, board) == move
        board.push(move)


def test_index_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        index_to_move(-1, chess.Board())
    with pytest.raises(ValueError):
        index_to_move(ACTION_SIZE, chess.Board())
