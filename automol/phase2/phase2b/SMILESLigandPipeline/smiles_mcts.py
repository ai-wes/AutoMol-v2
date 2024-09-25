import numpy as np

class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search."""

    def __init__(self, smiles, parent=None):
        self.smiles = smiles
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def is_fully_expanded(self):
        """Check if the node is fully expanded."""
        # Assuming each node can have up to 5 children
        return len(self.children) >= 5

    def best_child(self, c_param=1.4):
        """Select the best child based on UCB1."""
        choices_weights = [
            (child.reward / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]