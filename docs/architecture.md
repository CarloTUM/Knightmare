# Architecture

## Network

`PolicyValueNet` is a residual tower with optional squeeze-and-excitation
gating, followed by two heads:

- **Policy head**: a `1 x 1` convolution to 73 planes, flattened to 4672
  logits. The flattening order matches the AlphaZero action encoding so
  that the convolutional spatial structure aligns with the action layout.
- **Value head**: a `1 x 1` convolution to 32 planes, flattened, and run
  through a two-layer MLP with `tanh` output.

The default configuration (`128` filters, `10` blocks, SE on) totals
roughly 4 M parameters — small enough to train on a single GPU.

## Encoding

`board_to_tensor` produces an `(17, 8, 8)` tensor:

| Plane | Content                              |
|-------|--------------------------------------|
| 0–5   | White P, N, B, R, Q, K (one-hot)     |
| 6–11  | Black pieces                         |
| 12    | Side-to-move indicator               |
| 13–16 | Castling rights (WK, WQ, BK, BQ)     |
| 17    | Halfmove clock / 100                 |

Optionally `flip_for_black=True` mirrors the board so the side to move is
always at the bottom — this is the AlphaZero trick that lets the network
share representation across colours.

`encoding.move_to_index` / `index_to_move` implement the AlphaZero
73-plane action encoding:

- **0–55** queen moves (`8 directions × 1..7 squares`).
- **56–63** knight moves.
- **64–72** underpromotions (`{N, B, R} × {capture-left, push, capture-right}`).

Queen promotions reuse the queen-move planes; the decoder upgrades a
pawn move that lands on the back rank to a queen promotion automatically.

## Search

`MCTS` is a standard PUCT search with three production-flavoured
extensions:

1. **Dirichlet noise at the root** ensures self-play games explore
   alternatives rather than collapsing onto the network's current
   favourite.
2. **Virtual loss** lets the search descend several leaves before the
   network is invoked, so we can amortise GPU launches.
3. **Batched leaf evaluation**: up to `EVAL_BATCH` boards are stacked
   into one network call.

The search reuses the subtree across plies (`MCTS.advance`) so a
self-play game does not throw away the work of the previous move.

## Training

`train.train` iterates HDF5 shards through `DataLoader` workers, applies
mixed-precision forward / backward passes, and updates an EMA shadow
copy of the parameters. The loss is `MSE(value) + cross-entropy(policy)`,
the latter computed against the target distribution produced by the
search rather than a one-hot move.

A cosine learning-rate schedule with linear warm-up is used by default.
Checkpoints are written atomically (`.tmp` then rename) so a crash
during `torch.save` cannot corrupt the previous best model.
