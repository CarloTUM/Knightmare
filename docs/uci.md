# UCI engine

Knightmare ships a minimal UCI front-end (`knightmare uci` or the
`knightmare-uci` script). The engine is suitable for tournament play
in:

- [Arena](http://www.playwitharena.de/)
- [cutechess-cli](https://github.com/cutechess/cutechess) — useful for
  Elo gauntlet matches.
- [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) — to
  run as a Lichess bot account.

## cutechess-cli example

```bash
cutechess-cli \
    -engine name=Knightmare cmd=knightmare arg="uci --weights models/best_model.pth --sims 200" \
    -engine name=Stockfish cmd=stockfish \
    -each tc=40/60 -rounds 20 -concurrency 2 -pgnout games.pgn
```

## Supported UCI commands

`uci`, `isready`, `ucinewgame`, `position [startpos|fen ...] [moves ...]`,
`go [movetime|nodes|depth|wtime|btime]`, `setoption`, `stop`, `quit`.

The engine exposes three options:

| Option   | Type    | Default | Description                                  |
|----------|---------|---------|----------------------------------------------|
| Weights  | string  | `<none>`| Path to the network checkpoint (`.pth`).     |
| Sims     | spin    | `800`   | MCTS simulations per move.                   |
| Device   | string  | `cpu`   | `cpu` or `cuda` (or `cuda:0`, `cuda:1`, …).  |

## Time controls

`go movetime <ms>` and the side-to-move clocks `wtime` / `btime` are
mapped to a coarse simulation budget: 100 sims / second at a 1/30th time
slice. For accurate time management plug Knightmare into a wrapper that
sets a fixed `--sims` instead.
