import torch as torch
import torch.nn as nn
from typing import Optional
from math import floor, log2, sqrt
import torch.nn.functional as F

INIT_STRAT = 'hyperspherical-shell'

class FFF(nn.Module):
    def __init__(self, nIn: int, nOut: int, depth: Optional[int] = None):
        super().__init__()
        self.depth = depth or int(floor(log2(nIn)))  # depth is the number of decision boundaries
        nNodes = 2 ** self.depth - 1

        # each node "holds" a basis-vector in INPUT space (.X) and in OUTPUT space (.Y)

        if INIT_STRAT == 'gaussian':
            # This from orig authors; scaling looks off for self.Y
            def create_basis_vectors_of(length, scaling):
                return nn.Parameter(torch.empty(nNodes, length).uniform_(-scaling, scaling))
            self.X = create_basis_vectors_of(length=nIn, scaling=1/sqrt(nIn))
            self.Y = create_basis_vectors_of(length=nOut, scaling=1/sqrt(self.depth + 1))

        elif INIT_STRAT == 'hyperspherical-shell':
            # Initialize vectors on INPUT/OUTPUT space unit hypersphere
            #   (idea: basis vectors should be of unit length).
            def create_random_unit_vectors_of(length):
                weights = torch.randn(nNodes, length)  # Initialize weights randomly
                weights = F.normalize(weights, p=2, dim=-1)  # L2-Normalize along the last dimension
                return nn.Parameter(weights)
            self.X = create_random_unit_vectors_of(length=nIn)
            self.Y = create_random_unit_vectors_of(length=nOut)

    def forward(self, x: torch.Tensor):
        nBatch, nIn, nOut = x.shape[0], self.X.shape[-1], self.Y.shape[-1]

        current_node = torch.zeros(nBatch, dtype=torch.long, device=x.device)

        # Walk the tree, assembling y piecemeal
        y = torch.zeros((nBatch, nOut), dtype=torch.float, device=x.device)
        for _ in range(self.depth):
            # Project x onto the current node's INPUT basis vector
            #   λ = x DOT currNode.X
            # (nBatch, nIn,) (nBatch, nIn) -> (nBatch,)
            λ = torch.einsum("b i, b i -> b", x, self.X[current_node])

            # Project this contribution into OUTPUT space:
            #   y += λ currNode.Y
            # (nBatch,) (nBatch, nOut) -> (nBatch, nOut)
            y += torch.einsum("b, b j -> b j", λ, self.Y[current_node])

            # We'll branch right if x is "sunny-side" of the
            # hyperplane defined by node.x (else left)
            branch_choice = (λ > 0).long()

            # figure out index of node in next layer to visit
            current_node = (current_node * 2) + 1 + branch_choice

        return y

    def __repr__(self):
        return f"FFF({self.X.shape[-1]}, {self.Y.shape[-1]}, depth={self.depth})"
