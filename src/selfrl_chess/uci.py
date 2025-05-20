"""Minimal UCI protocol so Knightmare can run inside Arena, cutechess-cli, or
lichess-bot. The implementation is deliberately tiny -- just enough to be a
well-behaved engine for tournament play.

Supported commands:

    uci, isready, ucinewgame, position [startpos|fen ...] [moves ...],
    go [movetime <ms>|nodes <n>|depth <n>|wtime|btime|...], stop, quit, setoption.

The ``go`` handler always uses MCTS with a fixed simulation budget mapped
from the UCI time controls.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable

import chess
import numpy as np
import torch

from . import __version__
from .config import NUM_SIMULATIONS
from .mcts import MCTS
from .network import PolicyValueNet
from .utils import load_checkpoint

log = logging.getLogger(__name__)


@dataclass
class EngineOptions:
    weights: Path | None = None
    sims: int = NUM_SIMULATIONS
    device: str = "cpu"
    threads: int = 1


def _build_searcher(opts: EngineOptions) -> MCTS:
    net = PolicyValueNet().to(opts.device)
    if opts.weights is not None and opts.weights.exists():
        load_checkpoint(opts.weights, model=net, map_location=opts.device)
    net.eval()
    return MCTS(net, num_simulations=opts.sims, device=opts.device)


class UCIEngine:
    """Tiny single-threaded UCI engine.

    Communication is line-based on stdin/stdout; ``go`` runs synchronously,
    which is sufficient for engines whose search budget is bounded by a
    small fixed simulation count.
    """

    def __init__(
        self,
        opts: EngineOptions | None = None,
        *,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
        searcher_factory: Callable[[EngineOptions], MCTS] | None = None,
    ) -> None:
        self.opts = opts or EngineOptions()
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.board = chess.Board()
        self._searcher_factory = searcher_factory or _build_searcher
        self._searcher: MCTS | None = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------- lifecycle

    def run(self) -> None:
        for raw in self.stdin:
            line = raw.strip()
            if not line:
                continue
            if line == "quit":
                return
            try:
                self._handle(line)
            except Exception:  # pragma: no cover - defensive
                log.exception("uci command failed: %s", line)

    # ------------------------------------------------------------------ I/O

    def _send(self, msg: str) -> None:
        self.stdout.write(msg + "\n")
        self.stdout.flush()

    # --------------------------------------------------------------- handlers

    def _handle(self, line: str) -> None:
        cmd, _, args = line.partition(" ")
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            return
        handler(args.strip())

    def _cmd_uci(self, _args: str) -> None:
        self._send(f"id name Knightmare {__version__}")
        self._send("id author Carlo Deutschmann")
        self._send("option name Weights type string default <none>")
        self._send("option name Sims type spin default 800 min 1 max 100000")
        self._send("option name Device type string default cpu")
        self._send("uciok")

    def _cmd_isready(self, _args: str) -> None:
        with self._lock:
            if self._searcher is None:
                self._searcher = self._searcher_factory(self.opts)
        self._send("readyok")

    def _cmd_ucinewgame(self, _args: str) -> None:
        self.board = chess.Board()
        if self._searcher is not None:
            self._searcher.reset()

    def _cmd_position(self, args: str) -> None:
        tokens = args.split()
        if not tokens:
            return
        if tokens[0] == "startpos":
            self.board = chess.Board()
            rest = tokens[1:]
        elif tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            self.board = chess.Board(fen)
            rest = tokens[7:]
        else:
            return
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                self.board.push(chess.Move.from_uci(uci))
        if self._searcher is not None:
            self._searcher.reset()

    def _cmd_setoption(self, args: str) -> None:
        # "name X value Y"
        parts = args.split()
        if "name" not in parts or "value" not in parts:
            return
        try:
            ni = parts.index("name")
            vi = parts.index("value")
        except ValueError:
            return
        name = " ".join(parts[ni + 1 : vi]).lower()
        value = " ".join(parts[vi + 1 :])
        if name == "weights":
            self.opts.weights = Path(value) if value not in ("", "<none>") else None
            self._searcher = None
        elif name == "sims":
            self.opts.sims = max(1, int(value))
            self._searcher = None
        elif name == "device":
            self.opts.device = value
            self._searcher = None

    def _cmd_go(self, args: str) -> None:
        # We honour ``movetime``, ``nodes``, and the side-to-move clocks
        # ``wtime`` / ``btime`` in a coarse way; everything else just uses
        # the configured fixed simulation budget.
        tokens = args.split()
        sims = self.opts.sims
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "nodes" and i + 1 < len(tokens):
                sims = max(1, int(tokens[i + 1]))
                i += 2
                continue
            if tok == "movetime" and i + 1 < len(tokens):
                # Roughly 100 sims / 1000 ms on CPU.
                sims = max(1, int(int(tokens[i + 1]) / 10))
                i += 2
                continue
            if tok in {"wtime", "btime"} and i + 1 < len(tokens):
                # Use 1/30th of the remaining clock, capped at 5000 ms.
                ms_left = int(tokens[i + 1])
                if (tok == "wtime") == (self.board.turn == chess.WHITE):
                    sims = max(1, int(min(ms_left / 30, 5000) / 10))
                i += 2
                continue
            i += 1

        if self._searcher is None:
            self._searcher = self._searcher_factory(self.opts)
        self._searcher.num_simulations = sims

        t0 = time.perf_counter()
        pi = self._searcher.policy(self.board, temperature=1e-3, add_noise=False)
        idx = int(np.argmax(pi))
        from .encoding import index_to_move

        move = index_to_move(idx, self.board)
        if move not in self.board.legal_moves:
            move = next(iter(self.board.legal_moves))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        self._send(f"info depth 1 nodes {sims} time {max(1, elapsed_ms)}")
        self._send(f"bestmove {move.uci()}")


def main(argv: list[str] | None = None) -> int:
    opts = EngineOptions()
    if argv:
        # ``knightmare uci --weights path.pth --sims 400`` style.
        argv = list(argv)
        while argv:
            tok = argv.pop(0)
            if tok == "--weights" and argv:
                opts.weights = Path(argv.pop(0))
            elif tok == "--sims" and argv:
                opts.sims = int(argv.pop(0))
            elif tok == "--device" and argv:
                opts.device = argv.pop(0)
    UCIEngine(opts).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
