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


def create_uniform_weights(
    n_nodes: int, width: int, num_trees: Optional[int] = None
) -> torch.Tensor:
    """Initialize weights uniformly for each vector hold in nodes.

    Args:
        n_nodes (int): number of nodes in each tree
        width (int): dimension of vectors hold in each node
        num_trees (Optional[int], optional): number of trees in the forest
        init_factor (float): boundaries for uniform weight initialization
        Defaults to None - meaning that we have a single tree.

    Returns:
        torch.Tensor: _description_
    """
    init_factor = 1 / sqrt(width)
    if num_trees:
        return torch.empty(n_nodes, num_trees, width).uniform_(
            -init_factor, init_factor
        )
    else:
        return torch.empty(n_nodes, width).uniform_(-init_factor, init_factor)


def create_random_unit_vectors(
    n_nodes: int, width: int, num_trees: Optional[int] = None
) -> torch.Tensor:
    """Initialize weights randomly and then apply L2-Normalize along the last dimension.
    Args:
        n_nodes (int): number of nodes in each tree
        width (int): dimension of vectors hold in each node
        num_trees (Optional[int], optional): number of trees in the forest
        Defaults to None - meaning that we have a single tree

    Returns:
        torch.Tensor: _description_
    """
    if num_trees:
        weights = torch.randn(n_nodes, num_trees, width)
        weights = F.normalize(weights, p=2, dim=-1)
    return weights


class F4(nn.Module):
    """
    FFFF, where last F stands for Forest. Similar to FFF but runs multiple trees $T_k$ in parallel.
    Optionally accepts a normalization function acting on a set of $\lambda^t$ weights from
    each Tree on particular depth `t` $T^t_k$ - thus instead of "raw" $\lambda^t_k$ we use
    something like $\lambda^t_k}{\sqrt{\sum_{k=1}^K (\lambda^t_k)^2}} in case of `L2` normalization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_trees: int,
        depth: Optional[int] = None,
        normalize_func: Optional[callable] = None,
        init_strategy: Optional[str] = None,
        use_depth_fair: bool = False,
    ) -> torch.Tensor:
        """Note: softmax normalization is not recommended, because it will always give positive sign
        of lambdas and thus our branch selection to the right node.

        Args:
            in_features (int): number of features in the input, similar to `nn.Linear` argument.
            out_features (int): number of features in the output.
            num_trees (int): number of trees in that will will be run through in parallel.
                 Note that the final Basis constructed from the `output` vectors stored in each tree node
                 will (almost always) contain a mixed selection of vectors from different trees.
            depth (Optional[int], None): fixed depth of a single Tree.
                Note that since the number leaves grows as $2^depth$ in may be memory-costly to choose `depth` > 14.
                Defaults to None, which auto-selects `depth` as `log_2(in_features)`
            normalize_func (Optional[callable], None): function that performs normalization between
                vector associated $\lambda^t_k$ from each tree $T_k$ on each depth $t$
                Defaults to None (no normalization), F.norm
            init_strategy (Optional[str], None): one of 'hyperspherical-shell', 'gaussian', 'uniform', 'xavier'
            use_depth_fair (bool): used only with `depth == None`, mostly for benchmarking purposes,
                when `True` will substract `num_trees` from `depth` to equalize number of nodes with FFF

        Returns:
            torch.Tensor: _description_
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_trees = num_trees

        # depth is the number of decision boundaries
        if not depth:
            self.depth = int(floor(log2(in_features)))
            if use_depth_fair:
                self.depth = self.depth - self.num_trees
        else:
            self.depth = depth

        self.n_nodes = 2**self.depth - 1

        self.init_strategy = init_strategy or "hyperspherical-shell"

        self.keys = self.create_weights_parameters(
            num_features=self.in_features, init_strategy=self.init_strategy
        )
        self.values = self.create_weights_parameters(
            num_features=self.out_features, init_strategy=self.init_strategy
        )

        self.normalize: Optional[callable] = normalize_func

    def create_weights_parameters(
        self, num_features: int, init_strategy: str
    ) -> nn.Parameter:
        if init_strategy == "gaussian":
            tensor = create_uniform_weights(
                n_nodes=self.n_nodes, width=num_features, num_trees=self.num_trees
            )
        elif init_strategy == "hyperspherical-shell":
            tensor = create_random_unit_vectors(
                n_nodes=self.n_nodes, width=num_features, num_trees=self.num_trees
            )
        elif init_strategy == "xavier":
            pass
        else:
            raise NotImplementedError(
                f"weight init strategy {init_strategy} not supported!"
            )
        return nn.Parameter(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts tensor of size [B, `input_features`]
        Outputs tensor of size [B, `output_features`]
        Thus `num_trees` does not change output dimensionality,
        but allows our input to navigate multiple trees at once.
        """
        batch_size = x.shape[0]
        current_nodes = torch.zeros(batch_size, self.num_trees, dtype=torch.long)
        tree_selector = torch.arange(self.num_trees, dtype=torch.long)
        y = torch.zeros((batch_size, self.out_features), dtype=torch.float)

        # walk multiple trees at once, dynamically constructing a basis
        # each coeff will determine whether we branch left or right inside one tree
        # but also what is the chance to stay on that tree for the next depth iteration.
        # assemble our output `y` in a slow `for` loop
        # below `b` - batchsize, `k` - tree in the forest index, `i` - features dim
        for i in range(self.depth):
            lambda_ = torch.einsum(
                "bi, bki -> bk", x, self.keys[current_nodes, tree_selector]
            )

            # optional normalization
            # Note: softmax would always give positive lambdas / branch to the right
            # and thus "kill" part of the tree, use L2
            if self.normalize:
                lambda_ = self.normalize(lambda_)

            y += torch.einsum(
                "bk, bkj -> bj", lambda_, self.values[current_nodes, tree_selector]
            )

            # choose how to select branch for all `k` trees at once
            # branch right if  `x` projection on `k` node `key` is positive
            # i.e. `x` lies on the right side of the hyperplane orthogonal to `key`
            plane_choice = (lambda_ > 0).long()

            # figure out index of node in next layer to visit
            current_nodes = (current_nodes * 2) + 1 + plane_choice

        return y

    def __repr__(self):
        return f"FFFF({self.in_features}, {self.out_features}, num_trees={self.num_trees}, depth={self.depth})"
