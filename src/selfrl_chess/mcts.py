"""Monte-Carlo Tree Search with PUCT, Dirichlet noise and virtual loss.

The search follows the AlphaZero formulation. Highlights compared to a naive
implementation:

* Root nodes get Dirichlet noise mixed into their priors so self-play
  generates exploratory games.
* Virtual loss lets multiple simulations descend in flight without re-visiting
  the same path; useful even in the single-process case for batched leaf
  evaluation.
* Leaf evaluation is batched: the search collects up to ``EVAL_BATCH`` leaves
  before invoking the network, amortising GPU launch overhead.
* Terminal positions are scored exactly (+/- 1 / 0) instead of being passed
  through the network.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from .board import board_to_tensor
from .config import (
    CPUCT,
    DIRICHLET_ALPHA,
    DIRICHLET_EPS,
    EVAL_BATCH,
    NUM_SIMULATIONS,
    VIRTUAL_LOSS,
)
from .encoding import move_to_index
from .network import PolicyValueNet


@dataclass
class TreeNode:
    """One node of the search tree."""

    parent: TreeNode | None
    prior: float
    move: chess.Move | None = None
    visits: int = 0
    total_value: float = 0.0
    virtual_loss: float = 0.0
    children: dict[chess.Move, TreeNode] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_value: float = 0.0

    @property
    def q(self) -> float:
        if self.visits == 0:
            return 0.0
        return (self.total_value - self.virtual_loss) / max(self.visits, 1)

    def is_expanded(self) -> bool:
        return bool(self.children) or self.is_terminal

    def select(self, c_puct: float) -> tuple[chess.Move, TreeNode]:
        sqrt_n = math.sqrt(max(self.visits, 1))
        best_score = -float("inf")
        best_move: chess.Move | None = None
        best_child: TreeNode | None = None
        for move, child in self.children.items():
            u = c_puct * child.prior * sqrt_n / (1 + child.visits)
            score = child.q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        assert best_move is not None and best_child is not None
        return best_move, best_child

    def expand(self, priors: Iterable[tuple[chess.Move, float]]) -> None:
        for move, p in priors:
            if move not in self.children:
                self.children[move] = TreeNode(parent=self, prior=p, move=move)

    def backup(self, value: float) -> None:
        node: TreeNode | None = self
        sign = 1.0
        while node is not None:
            node.visits += 1
            node.total_value += value * sign
            sign = -sign
            node = node.parent


def _terminal_value(board: chess.Board) -> float | None:
    """Return the value (from the side-to-move perspective) for a terminal node."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class MCTS:
    """Batched PUCT search."""

    def __init__(
        self,
        net: PolicyValueNet,
        *,
        num_simulations: int = NUM_SIMULATIONS,
        c_puct: float = CPUCT,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_eps: float = DIRICHLET_EPS,
        eval_batch: int = EVAL_BATCH,
        virtual_loss: float = VIRTUAL_LOSS,
        device: torch.device | str = "cpu",
    ) -> None:
        self.net = net
        self.device = torch.device(device)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.eval_batch = max(1, eval_batch)
        self.virtual_loss = virtual_loss
        self.root: TreeNode | None = None

    # ------------------------------------------------------------------ public

    def reset(self) -> None:
        self.root = None

    def run(
        self,
        board: chess.Board,
        *,
        add_noise: bool = True,
    ) -> TreeNode:
        """Run ``num_simulations`` simulations rooted at ``board`` and return the root."""
        fresh_root = self.root is None
        if fresh_root:
            self.root = TreeNode(parent=None, prior=1.0)
            self._expand_root(board.copy(stack=False), self.root)

        if add_noise and fresh_root and self.root is not None and self.root.children:
            self._inject_dirichlet(self.root)

        sims_done = 0
        while sims_done < self.num_simulations:
            batch: list[tuple[TreeNode, chess.Board]] = []
            seen: set[int] = set()
            attempts = 0
            max_attempts = self.eval_batch * 4
            while (
                len(batch) < self.eval_batch
                and sims_done + len(batch) < self.num_simulations
                and attempts < max_attempts
            ):
                attempts += 1
                leaf, leaf_board = self._descend(board.copy(stack=False))
                term = _terminal_value(leaf_board)
                if term is not None:
                    leaf.is_terminal = True
                    leaf.terminal_value = term
                    self._undo_virtual_loss(leaf)
                    leaf.backup(term)
                    sims_done += 1
                    continue
                if id(leaf) in seen:
                    # Same leaf re-selected within this batch -- undo virtual
                    # loss on the path and try again so we do not double-count.
                    self._undo_virtual_loss(leaf)
                    continue
                seen.add(id(leaf))
                batch.append((leaf, leaf_board))
            if not batch:
                continue
            self._expand_and_backup(batch)
            sims_done += len(batch)

        assert self.root is not None
        return self.root

    def policy(
        self,
        board: chess.Board,
        *,
        temperature: float,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Return a length-4672 probability vector over actions."""
        from .encoding import ACTION_SIZE

        root = self.run(board, add_noise=add_noise)
        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move, child in root.children.items():
            visits[move_to_index(move)] = child.visits
        if visits.sum() == 0:
            visits[:] = 1.0 / ACTION_SIZE
            return visits
        if temperature <= 1e-3:
            best = int(np.argmax(visits))
            probs = np.zeros_like(visits)
            probs[best] = 1.0
            return probs
        scaled = visits ** (1.0 / temperature)
        return scaled / scaled.sum()

    def advance(self, move: chess.Move) -> None:
        """Move the root down to ``move`` so search trees can be reused across plies."""
        if self.root is None or move not in self.root.children:
            self.root = None
            return
        self.root = self.root.children[move]
        self.root.parent = None

    # ----------------------------------------------------------------- helpers

    def _descend(self, board: chess.Board) -> tuple[TreeNode, chess.Board]:
        assert self.root is not None
        node = self.root
        while node.is_expanded() and not node.is_terminal:
            move, node = node.select(self.c_puct)
            board.push(move)
            node.virtual_loss += self.virtual_loss
        return node, board

    def _undo_virtual_loss(self, node: TreeNode) -> None:
        cur: TreeNode | None = node
        while cur is not None and cur.parent is not None:
            cur.virtual_loss = max(0.0, cur.virtual_loss - self.virtual_loss)
            cur = cur.parent

    def _expand_root(self, board: chess.Board, node: TreeNode) -> None:
        priors, _ = self._evaluate([board])
        node.expand(priors[0])

    def _inject_dirichlet(self, node: TreeNode) -> None:
        if not node.children:
            return
        rng = np.random.default_rng()
        noise = rng.dirichlet([self.dirichlet_alpha] * len(node.children))
        eps = self.dirichlet_eps
        for (_, child), n in zip(node.children.items(), noise, strict=True):
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _expand_and_backup(self, batch: list[tuple[TreeNode, chess.Board]]) -> None:
        boards = [b for _, b in batch]
        priors_per_leaf, values = self._evaluate(boards)
        for (leaf, _board), priors, value in zip(batch, priors_per_leaf, values, strict=True):
            leaf.expand(priors)
            self._undo_virtual_loss(leaf)
            leaf.backup(value)

    @torch.inference_mode()
    def _evaluate(
        self, boards: list[chess.Board]
    ) -> tuple[list[list[tuple[chess.Move, float]]], list[float]]:
        tensors = np.stack([board_to_tensor(b) for b in boards])
        x = torch.from_numpy(tensors).to(self.device, non_blocking=True)
        log_probs, values = self.net(x)
        probs = log_probs.exp().cpu().numpy()
        values_list = values.view(-1).cpu().numpy().tolist()

        priors_per_leaf: list[list[tuple[chess.Move, float]]] = []
        for board, probs_row in zip(boards, probs, strict=True):
            legal = list(board.legal_moves)
            if not legal:
                priors_per_leaf.append([])
                continue
            indices = np.fromiter(
                (move_to_index(m) for m in legal), dtype=np.int64, count=len(legal)
            )
            masked = probs_row[indices]
            total = float(masked.sum())
            masked = np.full_like(masked, 1.0 / len(legal)) if total <= 1e-8 else masked / total
            priors_per_leaf.append(list(zip(legal, masked.astype(float).tolist(), strict=True)))
        return priors_per_leaf, values_list
