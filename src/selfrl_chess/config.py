"""Centralised configuration.

Most values come from ``KNIGHTMARE_*`` environment variables when set, which
makes it easy to override hyperparameters from CI or container deployments
without editing source.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import torch


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


ROOT_DIR: Final = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Final = _env_path("KNIGHTMARE_DATA_DIR", ROOT_DIR / "data")
MODELS_DIR: Final = _env_path("KNIGHTMARE_MODELS_DIR", ROOT_DIR / "models")
LOG_DIR: Final = _env_path("KNIGHTMARE_LOG_DIR", ROOT_DIR / "runs")

# MCTS
NUM_SIMULATIONS: Final = _env_int("KNIGHTMARE_SIMS", 800)
CPUCT: Final = _env_float("KNIGHTMARE_CPUCT", 1.5)
DIRICHLET_ALPHA: Final = _env_float("KNIGHTMARE_DIRICHLET_ALPHA", 0.3)
DIRICHLET_EPS: Final = _env_float("KNIGHTMARE_DIRICHLET_EPS", 0.25)
VIRTUAL_LOSS: Final = _env_float("KNIGHTMARE_VIRTUAL_LOSS", 1.0)
EVAL_BATCH: Final = _env_int("KNIGHTMARE_EVAL_BATCH", 16)

# Replay buffer / training
BUFFER_SIZE: Final = _env_int("KNIGHTMARE_BUFFER", 200_000)
BATCH_SIZE: Final = _env_int("KNIGHTMARE_BATCH", 512)
LEARNING_RATE: Final = _env_float("KNIGHTMARE_LR", 1e-3)
WEIGHT_DECAY: Final = _env_float("KNIGHTMARE_WEIGHT_DECAY", 1e-4)
EPOCHS: Final = _env_int("KNIGHTMARE_EPOCHS", 5)
GRAD_CLIP: Final = _env_float("KNIGHTMARE_GRAD_CLIP", 1.0)
EMA_DECAY: Final = _env_float("KNIGHTMARE_EMA_DECAY", 0.999)

# Self-play schedule
SELFPLAY_GAMES_PER_ITER: Final = _env_int("KNIGHTMARE_SELFPLAY_GAMES", 200)
SELFPLAY_TEMP_MOVES: Final = _env_int("KNIGHTMARE_TEMP_MOVES", 30)
RESIGN_THRESHOLD: Final = _env_float("KNIGHTMARE_RESIGN", -0.95)
MAX_GAME_PLIES: Final = _env_int("KNIGHTMARE_MAX_PLIES", 512)

# Network architecture
INPUT_PLANES: Final = 17
NUM_FILTERS: Final = _env_int("KNIGHTMARE_FILTERS", 128)
NUM_BLOCKS: Final = _env_int("KNIGHTMARE_BLOCKS", 10)
ACTION_SIZE: Final = 4672
USE_SE: Final = bool(_env_int("KNIGHTMARE_USE_SE", 1))

DEVICE: Final = os.environ.get(
    "KNIGHTMARE_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
SEED: Final = _env_int("KNIGHTMARE_SEED", 42)
