# Knightmare

An AlphaZero-style self-play reinforcement-learning chess engine written in
PyTorch. Knightmare ships a residual policy/value network with optional
squeeze-and-excitation gating, a batched PUCT Monte-Carlo Tree Search, a
self-play game generator, an Elo arena, a Lichess PGN ingestion pipeline,
and a UCI front-end so the engine can plug into chess GUIs (Arena,
cutechess-cli) and `lichess-bot`.

[![CI](https://github.com/CarloTUM/Knightmare/actions/workflows/ci.yml/badge.svg)](https://github.com/CarloTUM/Knightmare/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linted with ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Highlights

- **AlphaZero recipe**: residual tower + dual-head policy/value network,
  PUCT search with Dirichlet root noise, virtual-loss batched leaf
  evaluation, configurable temperature schedule.
- **AA+ project hygiene**: PEP 621 packaging, typed (`py.typed`), ruff +
  black + mypy, pytest with coverage, GitHub Actions CI on Python 3.10–3.12,
  pre-commit, dependabot, CodeQL, release workflow, semantic versioning,
  Conventional Commits.
- **Production-friendly training**: AMP (mixed precision), cosine LR with
  warm-up, AdamW + weight decay, gradient clipping, EMA shadow network,
  Tensorboard, atomic checkpoint writes.
- **Plays in real GUIs**: `knightmare uci` speaks UCI and runs in Arena,
  cutechess-cli, or lichess-bot.

## Installation

```bash
git clone https://github.com/CarloTUM/Knightmare.git
cd Knightmare
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For GPU training install the right PyTorch wheel for your CUDA toolkit
following <https://pytorch.org/get-started/locally/>.

## Quick start

```bash
# 1. Bootstrap from a Lichess PGN dump (optional but speeds up convergence).
knightmare ingest data/lichess_2024-01.pgn --out data/lichess.h5 --min-elo 2200

# 2. Train on whatever shards exist under data/.
knightmare train --epochs 5

# 3. Generate a batch of self-play games using the latest checkpoint.
knightmare selfplay --weights models/best_model.pth --games 200 --sims 400

# 4. Re-train -- the new shard joins the training set automatically.
knightmare train

# 5. Sanity-check progress against the previous best.
knightmare eval models/best_model.pth models/checkpoint_epoch4.pth --games 40

# 6. Plug into a GUI.
knightmare uci --weights models/best_model.pth --sims 400
```

## Architecture

```
selfrl_chess/
    board.py        Board -> 17-plane tensor.
    encoding.py     AlphaZero 73-plane action encoding (4672 actions).
    network.py      Residual tower + dual heads, with optional SE gating.
    mcts.py         Batched PUCT search with Dirichlet noise + virtual loss.
    selfplay.py     Self-play game generator -> HDF5 replay shards.
    replay.py       HDF5 replay-buffer reader/writer.
    train.py        AMP / cosine LR / EMA / Tensorboard training loop.
    eval.py         Arena that pits two checkpoints against each other.
    ingest.py       Lichess PGN -> HDF5 ingestion.
    uci.py          Minimal UCI engine front-end.
    cli.py          Typer CLI (`knightmare ...`).
    config.py       Hyperparameters; environment-variable overridable.
    utils.py        Logging, seeding, atomic checkpoint I/O.
    ema.py          Parameter EMA shadow network.
```

See [docs/architecture.md](docs/architecture.md) for a deeper tour of the
search, the encoding, and the training pipeline.

## Configuration

All hyperparameters in `selfrl_chess.config` can be overridden through
`KNIGHTMARE_*` environment variables — useful for sweep / CI runs:

```bash
KNIGHTMARE_SIMS=200 KNIGHTMARE_BATCH=256 knightmare train
```

## Development

```bash
make install      # editable install with dev extras
make lint         # ruff + black --check + mypy
make test         # pytest with coverage
make format       # black + ruff --fix
```

Pre-commit hooks (ruff, black, mypy, end-of-file fixer, etc.) run on every
commit:

```bash
pre-commit install
```

## Citation

If you use Knightmare in academic work, please cite the project — see
[CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE).
