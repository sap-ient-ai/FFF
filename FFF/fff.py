import torch as torch
import torch.nn as nn
from typing import Optional
from math import floor, log2, sqrt
import torch.nn.functional as F
from typing import Tuple

# The idea behind hyperspherical-shell initialization is that basis vectors
#   should be of unit length.
# Slightly less sure whether to init the OUTPUT basis vectors to _also_ be
#   of unit length, but they'll learn, so probably ok.
# Both strats seem to learn the same
INIT_STRAT = "hyperspherical-shell"  # 'spherical' or 'gaussian'


class FFF(nn.Module):
    def __init__(self, nIn: int, nOut: int, depth: Optional[int] = None):
        super().__init__()
        self.input_width = nIn
        self.output_width = nOut

        # depth is the number of decision boundaries
        self.depth = depth or int(floor(log2(nIn)))
        self.n_nodes = 2**self.depth - 1

        if INIT_STRAT == "gaussian":
            init_factor_I1 = 1 / sqrt(self.input_width)
            init_factor_I2 = 1 / sqrt(self.depth + 1)

            def create_weight_parameter(n_nodes, width, init_factor):
                return nn.Parameter(
                    torch.empty(n_nodes, width).uniform_(-init_factor, init_factor)
                )

            self.w1s = create_weight_parameter(self.n_nodes, nIn, init_factor_I1)
            self.w2s = create_weight_parameter(self.n_nodes, nOut, init_factor_I2)

        elif INIT_STRAT == "hyperspherical-shell":
            # each node has a w1 in INPUT space and a w2 in OUTPUT space
            # Initialize vectors on INPUT/OUTPUT space unit hypersphere
            def create_random_unit_vectors(n_nodes, width):
                weights = torch.randn(n_nodes, width)  # Initialize weights randomly
                weights = F.normalize(
                    weights, p=2, dim=-1
                )  # L2-Normalize along the last dimension
                return nn.Parameter(weights)

            self.w1s = create_random_unit_vectors(self.n_nodes, self.input_width)
            self.w2s = create_random_unit_vectors(self.n_nodes, self.output_width)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        current_node = torch.zeros(batch_size, dtype=torch.long)
        y = torch.zeros((batch_size, self.output_width), dtype=torch.float)

        # walk the tree, dynamically constructing a basis
        #   (and projection coeffs of x ONTO that basis)
        # Each coeff will determine whether we branch left or right
        #   as we move up the tree
        # We assemble our output y on the fly
        for i in range(self.depth):
            #
            # lambda_ = x DOT currNode.w1 (project x onto the current node's INPUT basis vector)
            lambda_ = torch.einsum("b i, b i -> b", x, self.w1s[current_node])

            # y += lambda_ * currNode.w2
            y += torch.einsum("b, bj -> bj", lambda_, self.w2s[current_node])

            # we'll branch right if x is "sunny-side" of the
            # hyperplane defined by node.x (else left)
            plane_choice = (lambda_ > 0).long()

            # figure out index of node in next layer to visit
            current_node = (current_node * 2) + 1 + plane_choice

        return y

    def __repr__(self):
        return f"FFF({self.input_width}, {self.output_width}, depth={self.depth})"
