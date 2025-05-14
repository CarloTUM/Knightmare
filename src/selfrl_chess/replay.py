"""HDF5 replay buffers used by self-play and supervised ingestion.

Layout of one shard ``shard.h5``:

    /states    (N, 17, 8, 8) float32
    /policies  (N, 4672)     float32  (target distribution)
    /values    (N,)          float32  (game outcome from the side to move)

Files are append-only; new self-play games extend the existing arrays via
chunked datasets, which keeps writes cheap and disk usage compact.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .encoding import ACTION_SIZE


def create_shard(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        f.create_dataset(
            "states",
            shape=(0, 17, 8, 8),
            maxshape=(None, 17, 8, 8),
            dtype="float32",
            chunks=(64, 17, 8, 8),
            compression="lzf",
        )
        f.create_dataset(
            "policies",
            shape=(0, ACTION_SIZE),
            maxshape=(None, ACTION_SIZE),
            dtype="float32",
            chunks=(64, ACTION_SIZE),
            compression="lzf",
        )
        f.create_dataset(
            "values",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            chunks=(1024,),
            compression="lzf",
        )
    return p


def append(
    path: str | Path,
    *,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
) -> int:
    """Append a batch of training examples and return the new size."""
    p = Path(path)
    if not p.exists():
        create_shard(p)
    if not (len(states) == len(policies) == len(values)):
        raise ValueError("states/policies/values must share the leading dim")
    with h5py.File(p, "a") as f:
        n_old = f["states"].shape[0]
        n_new = n_old + len(states)
        f["states"].resize((n_new, 17, 8, 8))
        f["policies"].resize((n_new, ACTION_SIZE))
        f["values"].resize((n_new,))
        f["states"][n_old:n_new] = states.astype("float32", copy=False)
        f["policies"][n_old:n_new] = policies.astype("float32", copy=False)
        f["values"][n_old:n_new] = values.astype("float32", copy=False)
        return n_new


@contextmanager
def open_shard(path: str | Path) -> Iterator[h5py.File]:
    p = Path(path)
    with h5py.File(p, "r") as f:
        yield f


def shard_size(path: str | Path) -> int:
    with open_shard(path) as f:
        return int(f["states"].shape[0])
