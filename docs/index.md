# Knightmare

> An AlphaZero-style self-play chess engine in PyTorch.

Knightmare is a clean-room reimplementation of the AlphaZero recipe for
chess: a deep residual policy/value network learns by playing against
itself, guided by a Monte-Carlo Tree Search that uses the network's
predictions as priors.

The codebase is small enough to read end-to-end in an afternoon, but
ships the kind of infrastructure that scales: HDF5 replay shards, mixed-
precision training, Tensorboard hooks, an Elo arena, a UCI engine
front-end, and a Typer CLI.

## Where to go next

- [Installation](installation.md) — get the package and dependencies.
- [Quickstart](quickstart.md) — five commands to a working engine.
- [Architecture](architecture.md) — the search, encoding and network in
  detail.
- [Training](training.md) — supervised bootstrap and self-play loop.
- [UCI engine](uci.md) — wire Knightmare into Arena, cutechess-cli, or
  lichess-bot.
- [API reference](api.md) — auto-generated from docstrings.
