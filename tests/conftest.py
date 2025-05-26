"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force CPU + a fixed seed so tests are reproducible."""
    monkeypatch.setenv("KNIGHTMARE_DEVICE", "cpu")
    monkeypatch.setenv("KNIGHTMARE_SEED", "0")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
