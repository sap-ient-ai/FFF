import math
from torch import nn
from torch.autograd import Function
import torch

if not torch.cuda.is_available():
    raise Exception("CUDA is not available")

from torch.utils.cpp_extension import load

import os

# get the absolute path of the current file
dir_path = os.path.dirname(os.path.realpath(__file__))

# REQUIREMENTS:
# - CUDA Toolkit 12.x (12.3 on my machine)
# - working C++ compiler (gcc 9 or greater, C++-11 on my machine) [MSVC compiler on Windows]
# - PyTorch 2.x with Cuda support
fff_cuda_ = load(
    "fff_cuda_",
    [dir_path + "/CUDA/fff_cuda.cpp", dir_path + "/CUDA/fff_kernel.cu"],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["--std=c++20", "--expt-relaxed-constexpr", "-O3"],
    extra_cflags=["--std=c++20", "-O3"],
)


from typing import Optional
from math import floor, log2, sqrt
import torch.nn.functional as F


class FFFLayer(nn.Module):
    def __init__(self, nIn: int, nOut: int, depth: Optional[int] = None):
        super().__init__()

        self.input_width = nIn
        self.output_width = nOut

        # depth is the number of decision boundaries
        self.depth = depth or int(floor(log2(nIn)))
        self.n_nodes = 2**self.depth - 1

        def create_random_unit_vectors(n_nodes, width):
            # Initialize weights randomly
            weights = torch.randn(n_nodes, width)
            # L2-Normalize along the last dimension
            weights = F.normalize(weights, p=2, dim=-1)
            return nn.Parameter(weights)

        self.w1s = create_random_unit_vectors(self.n_nodes, self.input_width)
        self.w2s = create_random_unit_vectors(self.n_nodes, self.output_width)

    def forward(self, input):
        return FFFFunction.apply(
            input,
            self.w1s,
            self.w2s,
            self.depth,
        )


class FFFFunction(Function):
    @staticmethod
    def forward(ctx, input, in_projection, out_projection, depth):
        # oldx has shape (..., width)
        # in_weight has shape (n_nodes, width)
        # out_weight has shape (n_nodes, width)
        result = fff_cuda_.forward(input, in_projection, out_projection, depth)
        # ctx.save_for_backward(input, in_projection, out_projection, depth)
        return result

    @staticmethod
    def backward(ctx, grad_of_output):
        raise NotImplementedError("Not implemented yet")
