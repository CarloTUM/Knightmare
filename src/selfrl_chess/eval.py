"""Arena: pit two checkpoints against each other and compute an Elo delta."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch

from .mcts import MCTS
from .network import PolicyValueNet
from .utils import load_checkpoint

log = logging.getLogger(__name__)


@dataclass
class ArenaResult:
    wins_a: int
    losses_a: int
    draws: int
    elo_delta: float

    @property
    def games(self) -> int:
        return self.wins_a + self.losses_a + self.draws

    @property
    def score_a(self) -> float:
        return (self.wins_a + 0.5 * self.draws) / max(1, self.games)


def _elo_delta(score: float) -> float:
    score = max(min(score, 1 - 1e-9), 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def _load(path: Path, *, device: str) -> PolicyValueNet:
    net = PolicyValueNet().to(device)
    load_checkpoint(path, model=net, map_location=device)
    net.eval()
    return net


def _play(white: MCTS, black: MCTS, *, max_plies: int) -> chess.Outcome | None:
    board = chess.Board()
    white.reset()
    black.reset()
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        searcher = white if board.turn == chess.WHITE else black
        pi = searcher.policy(board, temperature=1e-3, add_noise=False)
        idx = int(np.argmax(pi))
        from .encoding import index_to_move

        move = index_to_move(idx, board)
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)
        white.advance(move)
        black.advance(move)
    return board.outcome(claim_draw=True)


def arena(
    model_a: Path,
    model_b: Path,
    *,
    games: int = 40,
    sims: int = 100,
    max_plies: int = 512,
    device: str | torch.device = "cpu",
) -> ArenaResult:
    """Play ``games`` games (alternating colours) between two checkpoints."""
    net_a = _load(model_a, device=str(device))
    net_b = _load(model_b, device=str(device))

    wins = losses = draws = 0
    for game_idx in range(games):
        a_white = game_idx % 2 == 0
        if a_white:
            white = MCTS(net_a, num_simulations=sims, device=device)
            black = MCTS(net_b, num_simulations=sims, device=device)
        else:
            white = MCTS(net_b, num_simulations=sims, device=device)
            black = MCTS(net_a, num_simulations=sims, device=device)
        outcome = _play(white, black, max_plies=max_plies)

        if outcome is None or outcome.winner is None:
            draws += 1
            verdict = "draw"
        elif (outcome.winner == chess.WHITE) == a_white:
            wins += 1
            verdict = "A wins"
        else:
            losses += 1
            verdict = "B wins"
        log.info("game %d/%d: %s", game_idx + 1, games, verdict)

    score = (wins + 0.5 * draws) / max(1, wins + losses + draws)
    return ArenaResult(wins, losses, draws, _elo_delta(score))
