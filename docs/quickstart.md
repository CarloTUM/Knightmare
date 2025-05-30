# Quickstart

```bash
# 1. Optional: warm-start from a Lichess PGN dump.
knightmare ingest data/lichess_2024-01.pgn --out data/lichess.h5 --min-elo 2200

# 2. Train.
knightmare train --epochs 5

# 3. Generate self-play games.
knightmare selfplay --weights models/best_model.pth --games 200 --sims 400

# 4. Iterate.
knightmare train

# 5. Compare checkpoints.
knightmare eval models/best_model.pth models/checkpoint_epoch4.pth --games 40

# 6. Run as a UCI engine.
knightmare uci --weights models/best_model.pth --sims 400
```

The cycle is supervised bootstrap → train → self-play → train → arena.
A typical iteration on a single GPU takes a few hours and yields a
measurable Elo gain.
