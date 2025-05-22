"""Typer-based command-line interface.

Run ``knightmare --help`` for the full sub-command list. The CLI is the
recommended way to invoke training, self-play, evaluation, ingestion and the
UCI engine; the underlying functions are exposed for programmatic use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from . import __version__
from .config import (
    DATA_DIR,
    DEVICE,
    MODELS_DIR,
    NUM_SIMULATIONS,
    SELFPLAY_GAMES_PER_ITER,
)
from .utils import configure_logging

app = typer.Typer(
    name="knightmare",
    no_args_is_help=True,
    add_completion=False,
    help="Knightmare: AlphaZero-style chess engine.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"knightmare {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    version: bool = typer.Option(  # noqa: ARG001 -- consumed by callback
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.command("selfplay", help="Generate self-play games into an HDF5 shard.")
def cmd_selfplay(
    weights: Path = typer.Option(None, exists=False, help="Network checkpoint to use."),
    games: int = typer.Option(SELFPLAY_GAMES_PER_ITER, min=1),
    sims: int = typer.Option(NUM_SIMULATIONS, min=1),
    out_dir: Path = typer.Option(DATA_DIR),
    device: str = typer.Option(DEVICE),
) -> None:
    from .network import PolicyValueNet
    from .selfplay import run as selfplay_run
    from .utils import load_checkpoint

    net = PolicyValueNet().to(device)
    if weights is not None and weights.exists():
        load_checkpoint(weights, model=net, map_location=device)
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    selfplay_run(net, games=games, out_dir=out_dir, device=device, sims=sims)


@app.command("train", help="Train a network from existing replay shards.")
def cmd_train(
    data_dir: Path = typer.Option(DATA_DIR),
    epochs: int = typer.Option(None, min=1),
    batch_size: int = typer.Option(None, min=1),
    learning_rate: float = typer.Option(None),
    device: str = typer.Option(DEVICE),
) -> None:
    from .train import TrainConfig, train

    cfg = TrainConfig(device=device)
    if epochs:
        cfg.epochs = epochs
    if batch_size:
        cfg.batch_size = batch_size
    if learning_rate:
        cfg.learning_rate = learning_rate

    shards = sorted(Path(data_dir).glob("*.h5"))
    if not shards:
        raise typer.BadParameter(f"no shards found under {data_dir}")
    train(shards, cfg=cfg)


@app.command("eval", help="Pit two checkpoints against each other.")
def cmd_eval(
    a: Path = typer.Argument(..., exists=True),
    b: Path = typer.Argument(..., exists=True),
    games: int = typer.Option(40, min=2),
    sims: int = typer.Option(100, min=1),
    device: str = typer.Option(DEVICE),
) -> None:
    from .eval import arena

    res = arena(a, b, games=games, sims=sims, device=device)
    typer.echo(
        f"A wins={res.wins_a} B wins={res.losses_a} draws={res.draws} "
        f"score={res.score_a:.3f} elo_delta={res.elo_delta:+.1f}"
    )


@app.command("ingest", help="Ingest a Lichess PGN file into a replay shard.")
def cmd_ingest(
    pgn: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(DATA_DIR / "lichess.h5"),
    min_elo: int = typer.Option(2000, min=0),
) -> None:
    from .ingest import IngestFilter, ingest

    filt = IngestFilter(min_elo=min_elo)
    written = ingest(pgn, out_path=out, filt=filt)
    typer.echo(f"wrote {written} positions to {out}")


@app.command("uci", help="Run the engine in UCI protocol mode (for chess GUIs).")
def cmd_uci(
    weights: Path = typer.Option(None),
    sims: int = typer.Option(NUM_SIMULATIONS, min=1),
    device: str = typer.Option(DEVICE),
) -> None:
    from .uci import EngineOptions, UCIEngine

    UCIEngine(EngineOptions(weights=weights, sims=sims, device=device)).run()


@app.command("info", help="Print library and runtime information.")
def cmd_info() -> None:
    import torch

    typer.echo(f"knightmare {__version__}")
    typer.echo(f"torch {torch.__version__}")
    typer.echo(f"cuda {torch.cuda.is_available()}")
    typer.echo(f"device {DEVICE}")
    typer.echo(f"data_dir {DATA_DIR}")
    typer.echo(f"models_dir {MODELS_DIR}")


def main() -> None:
    app()
