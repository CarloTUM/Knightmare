# Training

The training pipeline supports two modes that compose into the
AlphaZero-style loop:

## Supervised bootstrap

Train against human Lichess games to get a useful starting policy:

```bash
knightmare ingest data/lichess_db_standard_2024-01.pgn \
    --out data/lichess.h5 --min-elo 2200
knightmare train --epochs 5
```

`ingest` filters by Elo and termination, drops engine games via the
`Termination` PGN header, and produces a streaming HDF5 shard.

## Self-play reinforcement learning

```bash
knightmare selfplay --weights models/best_model.pth --games 200 --sims 400
knightmare train
```

`selfplay` writes one shard per invocation; `train` consumes all shards
under the data directory.

## Tensorboard

If `tensorboard` is installed, training logs to `runs/`:

```bash
tensorboard --logdir runs
```

Tracked scalars: total loss, policy loss, value loss, learning rate.

## Hyperparameters

All hyperparameters live in `selfrl_chess.config`. Override at runtime
through environment variables, e.g.:

```bash
KNIGHTMARE_BATCH=256 KNIGHTMARE_LR=5e-4 knightmare train --epochs 8
```
