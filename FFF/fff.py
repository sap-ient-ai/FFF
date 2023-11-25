import torch as torch
import torch.nn as nn
from typing import Optional
from math import ceil, log2, sqrt


class FFF(nn.Module):
    def __init__(self, nIn: int, nOut: int, depth: Optional[int] = None):
        super().__init__()
        self.input_width = nIn
        self.output_width = nOut

        # depth is the number of decision boundaries
        self.depth = int(ceil(log2(nIn))) if depth is None else depth
        self.n_nodes = 2 ** (self.depth + 1) - 1

        self._initiate_weights()

    def _initiate_weights(self):
        init_factor_I1 = 1 / sqrt(self.input_width)
        init_factor_I2 = 1 / sqrt(self.depth + 1)

        # shape: (n_nodes, input_width)
        # weights for basis nodes projection
        self.w1s = nn.Parameter(
            torch.empty(self.n_nodes, self.input_width).uniform_(
                -init_factor_I1, init_factor_I1
            ),
            requires_grad=True,
        )

        # shape: (n_nodes, output_width)
        # weights for transforming basis nodes to output space
        self.w2s = nn.Parameter(
            torch.empty(self.n_nodes, self.output_width).uniform_(
                -init_factor_I2, init_factor_I2
            ),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # concurrent for batch size (bs, )
        current_node = torch.zeros((batch_size,), dtype=torch.long)

        all_nodes = torch.zeros(batch_size, self.depth + 1, dtype=torch.long)
        all_scores = torch.empty((batch_size, self.depth + 1), dtype=torch.float)

        # find the basis scalar for each node
        for i in range(self.depth + 1):
            # compute plane scores
            # dot product between input (x) and weights of the current node (w1s)
            # result is scalar of shape (bs)
            plane_score = torch.einsum("b i, b i -> b", x, self.w1s[current_node])
            all_nodes[:, i] = current_node

            # scores are used for gradient propagation and learning decision boundaries
            all_scores[:, i] = plane_score

            # compute next node (left or right)
            plane_choice = (plane_score > 0).long()
            current_node = (current_node * 2) + plane_choice + 1

        # reshape all_nodes to (bs, n_nodes, output_width)
        selected_w2s = self.w2s[all_nodes.flatten()].view(
            batch_size, self.depth + 1, self.output_width
        )

        # project scores to output space (bs, n_nodes, output_width), (bs, output_width) -> (bs, output_width)
        y = torch.einsum("b i j , b i -> b j", selected_w2s, all_scores)
        return y

    def __repr__(self):
        return f"FFF({self.input_width}, {self.output_width}, depth={self.depth})"
