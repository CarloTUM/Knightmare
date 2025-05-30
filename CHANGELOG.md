# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-30

### Added
- AlphaZero-style 73-plane move encoding (`encoding.py`, 4672 actions).
- Squeeze-and-excitation residual tower in `PolicyValueNet`.
- Batched PUCT MCTS with Dirichlet root noise and virtual loss.
- Self-play loop with temperature schedule and resignation threshold.
- HDF5 replay-buffer reader/writer with append-only chunked layout.
- AMP + cosine LR + EMA + Tensorboard training pipeline.
- Elo arena (`knightmare eval`) for checkpoint comparison.
- Lichess PGN ingestion (`knightmare ingest`).
- UCI engine front-end (`knightmare uci`).
- Typer-based CLI with sub-commands: `train`, `selfplay`, `eval`, `ingest`,
  `uci`, `info`.
- PEP 621 packaging, `py.typed` marker, ruff + black + mypy, pre-commit,
  GitHub Actions CI, CodeQL, dependabot, release workflow.
- mkdocs-material documentation site.
- Dockerfile (CPU + CUDA variants) and Makefile.

### Notes
- Versions before 0.1.0 were unreleased exploratory commits and do not
  carry stability guarantees.

[0.1.0]: https://github.com/CarloTUM/Knightmare/releases/tag/v0.1.0
