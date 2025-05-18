"""Convert Lichess PGN exports into HDF5 replay shards.

Useful as a warm-start: a network pre-trained on strong human games
converges far faster during self-play.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import numpy as np

from .board import board_to_tensor
from .encoding import ACTION_SIZE, move_to_index
from .replay import append, create_shard

log = logging.getLogger(__name__)


@dataclass
class IngestFilter:
    min_elo: int = 2000
    max_moves: int = 256
    require_termination: bool = True


def _eligible(game: chess.pgn.Game, filt: IngestFilter) -> bool:
    headers = game.headers
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    if min(white_elo, black_elo) < filt.min_elo:
        return False
    if filt.require_termination and headers.get("Termination", "Normal") != "Normal":
        return False
    return True


def _value_for_position(turn: bool, result: str) -> float:
    if result == "1/2-1/2":
        return 0.0
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if turn == chess.WHITE else 1.0
    return 0.0


def _iterate_examples(
    pgn_path: Path,
    filt: IngestFilter,
) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            if not _eligible(game, filt):
                continue
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= filt.max_moves:
                    break
                state = board_to_tensor(board)
                policy = np.zeros(ACTION_SIZE, dtype=np.float32)
                try:
                    policy[move_to_index(move)] = 1.0
                except ValueError:
                    board.push(move)
                    continue
                value = _value_for_position(board.turn, result)
                yield state, policy, value
                board.push(move)


def ingest(
    pgn_path: Path,
    *,
    out_path: Path,
    flush_every: int = 4096,
    filt: IngestFilter | None = None,
) -> int:
    """Stream ``pgn_path`` into ``out_path``; returns the number of positions written."""
    filt = filt or IngestFilter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        create_shard(out_path)

    buf_states: list[np.ndarray] = []
    buf_policies: list[np.ndarray] = []
    buf_values: list[float] = []
    written = 0

    def flush() -> None:
        nonlocal written
        if not buf_states:
            return
        append(
            out_path,
            states=np.stack(buf_states),
            policies=np.stack(buf_policies),
            values=np.asarray(buf_values, dtype=np.float32),
        )
        written += len(buf_states)
        buf_states.clear()
        buf_policies.clear()
        buf_values.clear()
        log.info("ingest progress: %d positions", written)

    for state, policy, value in _iterate_examples(pgn_path, filt):
        buf_states.append(state)
        buf_policies.append(policy)
        buf_values.append(value)
        if len(buf_states) >= flush_every:
            flush()
    flush()
    return written
