from __future__ import annotations

import io

import chess
import pytest

pytest.importorskip("torch")

from selfrl_chess.mcts import MCTS
from selfrl_chess.network import PolicyValueNet
from selfrl_chess.uci import EngineOptions, UCIEngine


def _engine(stdin: str) -> tuple[UCIEngine, io.StringIO]:
    """Build a UCI engine wired to in-memory streams and a tiny searcher."""
    out = io.StringIO()
    inp = io.StringIO(stdin)

    def factory(opts: EngineOptions) -> MCTS:
        net = PolicyValueNet(num_filters=4, num_blocks=1).eval()
        return MCTS(net, num_simulations=2, eval_batch=2)

    engine = UCIEngine(EngineOptions(), stdin=inp, stdout=out, searcher_factory=factory)
    return engine, out


def test_uci_handshake_emits_uciok_and_readyok() -> None:
    engine, out = _engine("uci\nisready\nquit\n")
    engine.run()
    text = out.getvalue()
    assert "uciok" in text
    assert "readyok" in text
    assert "id name Knightmare" in text


def test_uci_position_and_go_emit_legal_bestmove() -> None:
    engine, out = _engine(
        "uci\nisready\nucinewgame\nposition startpos\ngo nodes 2\nquit\n"
    )
    engine.run()
    lines = out.getvalue().splitlines()
    bestmove = next(line for line in lines if line.startswith("bestmove "))
    uci = bestmove.split()[1]
    move = chess.Move.from_uci(uci)
    assert move in chess.Board().legal_moves
