from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from selfrl_chess.encoding import ACTION_SIZE
from selfrl_chess.replay import append, create_shard, shard_size


def test_append_grows_shard(tmp_path: Path) -> None:
    shard = tmp_path / "replay.h5"
    create_shard(shard)
    states = np.zeros((3, 17, 8, 8), dtype=np.float32)
    policies = np.zeros((3, ACTION_SIZE), dtype=np.float32)
    values = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    new_size = append(shard, states=states, policies=policies, values=values)
    assert new_size == 3
    assert shard_size(shard) == 3
    new_size = append(shard, states=states, policies=policies, values=values)
    assert new_size == 6
    assert shard_size(shard) == 6


def test_append_validates_shape(tmp_path: Path) -> None:
    shard = tmp_path / "replay.h5"
    create_shard(shard)
    with pytest.raises(ValueError):
        append(
            shard,
            states=np.zeros((2, 17, 8, 8), dtype=np.float32),
            policies=np.zeros((3, ACTION_SIZE), dtype=np.float32),
            values=np.zeros((3,), dtype=np.float32),
        )
