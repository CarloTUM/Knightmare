import math
import numpy as np
import chess
import torch
from collections import defaultdict

# Absolute imports – greifen auf Module im selben Ordner zu
from network import PolicyValueNet
from config import CPUCT, NUM_SIMULATIONS, DEVICE
from board import board_to_tensor
from utils import move_to_index


class TreeNode:
    """
    Single node in the MCTS tree.
    """
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.P = prior_p    # prior probability
        self.N = 0          # visit count
        self.W = 0.0        # total value
        self.Q = 0.0        # mean value
        self.children = {}  # map action -> TreeNode

    def expand(self, action_priors):
        """
        Add new children for each legal move with given prior probability.
        action_priors: list of (move, prob)
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        Select action among children that maximizes PUCT score.
        Returns: (action, next_node)
        """
        return max(
            self.children.items(),
            key=lambda act_node: act_node[1]._ucb_score(c_puct)
        )

    def _ucb_score(self, c_puct):
        """
        Calculate and return the UCB score for this node.
        """
        u = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + u

    def update(self, leaf_value):
        """
        Update visit count, total and mean value.
        """
        self.N += 1
        self.W += leaf_value
        self.Q = self.W / self.N

    def update_recursive(self, leaf_value):
        """
        Update this node and all ancestors.
        leaf_value: evaluation from the viewpoint of the player who just moved.
        """
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


class MCTS:
    """
    Monte Carlo Tree Search with PUCT and neural network evaluation.
    """
    def __init__(self, policy_value_net: PolicyValueNet):
        self.root = TreeNode(None, 1.0)
        self.policy_value_net = policy_value_net.to(DEVICE)
        self.policy_value_net.eval()

    def _evaluate_leaf(self, board: chess.Board):
        """
        Run the neural network to obtain move priors and leaf value.
        Returns:
            action_priors: list of (move, prob) for legal moves
            leaf_value: float
        """
        state = board_to_tensor(board)
        state = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            log_probs, value = self.policy_value_net(state)
        probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
        leaf_value = value.item()

        legal_moves = list(board.legal_moves)
        action_priors = []
        prob_sum = 0.0
        for move in legal_moves:
            idx = move_to_index(move)
            p = probs[idx]
            action_priors.append((move, p))
            prob_sum += p

        action_priors = [(m, p / prob_sum) for m, p in action_priors]
        return action_priors, leaf_value

    def _simulate(self, board: chess.Board):
        """
        Perform one simulation from the root.
        """
        node = self.root
        moves_played = []

        # Selection
        while node.children:
            action, node = node.select(CPUCT)
            board.push(action)
            moves_played.append(action)

        # Expansion & Evaluation
        action_priors, leaf_value = self._evaluate_leaf(board)
        node.expand(action_priors)

        # Backpropagation
        node.update_recursive(-leaf_value)

        # Undo selected moves
        for _ in moves_played:
            board.pop()

    def get_move_probs(self, board: chess.Board, temp: float = 1e-3):
        """
        Run MCTS simulations and compute move probabilities.
        Returns:
            moves: list of chess.Move
            probs: numpy array of probabilities
        """
        self.root = TreeNode(None, 1.0)
        for _ in range(NUM_SIMULATIONS):
            board_copy = board.copy()
            self._simulate(board_copy)

        counts = [(child.N, action) for action, child in self.root.children.items()]
        visits = np.array([cnt for cnt, _ in counts], dtype=np.float32)

        if temp == 0:
            best = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[best] = 1.0
        else:
            visits = visits ** (1 / temp)
            probs = visits / np.sum(visits)

        moves = [action for _, action in counts]
        return moves, probs
