"""Self-play game generation.

A worker plays a complete game against itself with MCTS as the move
selector and writes the resulting (state, policy, value) triples to an
HDF5 shard. The temperature schedule follows AlphaZero: ``T=1`` for the
opening, then near-greedy thereafter so the engine consolidates around
its strongest moves.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

import chess
import numpy as np
import torch

from .board import board_to_tensor
from .config import (
    DATA_DIR,
    MAX_GAME_PLIES,
    RESIGN_THRESHOLD,
    SELFPLAY_GAMES_PER_ITER,
    SELFPLAY_TEMP_MOVES,
)
from .mcts import MCTS
from .network import PolicyValueNet
from .replay import append, create_shard

log = logging.getLogger(__name__)


def play_one_game(
    mcts: MCTS,
    *,
    temp_moves: int = SELFPLAY_TEMP_MOVES,
    max_plies: int = MAX_GAME_PLIES,
    resign_threshold: float = RESIGN_THRESHOLD,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Play a self-play game and return (states, policies, value_for_side_to_move)."""
    board = chess.Board()
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    side_to_move: list[bool] = []  # True = white moved at this ply
    mcts.reset()

    plies = 0
    resigned: bool | None = None
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        temperature = 1.0 if plies < temp_moves else 1e-3
        pi = mcts.policy(board, temperature=temperature)
        states.append(board_to_tensor(board))
        policies.append(pi)
        side_to_move.append(board.turn == chess.WHITE)

        # Resign if our root value estimate is hopeless.
        root_value = mcts.root.q if mcts.root is not None else 0.0
        if resign_threshold is not None and root_value < resign_threshold and plies > 30:
            resigned = board.turn == chess.WHITE
            break

        from .encoding import index_to_move

        if temperature <= 1e-3:
            move_idx = int(np.argmax(pi))
        else:
            move_idx = int(np.random.choice(len(pi), p=pi))
        move = index_to_move(move_idx, board)
        if move not in board.legal_moves:
            log.warning("MCTS returned illegal move %s, falling back", move.uci())
            move = next(iter(board.legal_moves))
        board.push(move)
        mcts.advance(move)
        plies += 1

    winner: bool | None
    if resigned is not None:
        winner = chess.BLACK if resigned else chess.WHITE
    else:
        outcome = board.outcome(claim_draw=True)
        winner = outcome.winner if outcome is not None else None

    if winner is None:
        values = [0.0 for _ in side_to_move]
    else:
        white_won = winner == chess.WHITE
        values = [1.0 if (m == white_won) else -1.0 for m in side_to_move]

    return states, policies, values


def run(
    net: PolicyValueNet,
    *,
    games: int = SELFPLAY_GAMES_PER_ITER,
    out_dir: Path = DATA_DIR,
    device: str | torch.device = "cpu",
    sims: int | None = None,
) -> Path:
    """Generate ``games`` self-play games and write them into a fresh shard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"replay_{int(time.time())}_{uuid.uuid4().hex[:6]}.h5"
    shard = out_dir / shard_name
    create_shard(shard)

    mcts_kwargs: dict[str, object] = {"device": device}
    if sims is not None:
        mcts_kwargs["num_simulations"] = sims
    mcts = MCTS(net, **mcts_kwargs)  # type: ignore[arg-type]

    total = 0
    for game_idx in range(games):
        t0 = time.perf_counter()
        states, policies, values = play_one_game(mcts)
        if states:
            append(
                shard,
                states=np.stack(states),
                policies=np.stack(policies),
                values=np.asarray(values, dtype=np.float32),
            )
        total += len(states)
        log.info(
            "game %d/%d plies=%d positions=%d wallclock=%.1fs",
            game_idx + 1,
            games,
            len(states),
            total,
            time.perf_counter() - t0,
        )
    log.info("self-play done: %d positions written to %s", total, shard)
    return shard


def iter_shards(out_dir: Path = DATA_DIR) -> Iterable[Path]:
    return sorted(out_dir.glob("replay_*.h5"))
