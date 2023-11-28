#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int w1s_sz, const int w2s_sz,
                               F *__restrict__ const _y,
                               F *__restrict__ const _x,
                               F *__restrict__ const _in_projection,
                               F *__restrict__ const _out_projection,
                               const unsigned int depth) {
  // compute which row and column of inputs we're dealing with
  const int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;

  // check if the row and column indices are valid
  if (row_index < B && col_index < w2s_sz) {
    // initialize the current node to zero
    int current_node = 0;
    // initialize the output vector to zero
    // F *y = _y + row_index * w2s_sz;
    F *y = _y + row_index * w2s_sz + col_index;
    F *x = _x + row_index * w1s_sz;

    // walk the tree, dynamically constructing a basis
    // and projection coeffs of x onto that basis
    // each coeff will determine whether we branch left or right
    // as we move up the tree
    // we assemble our output y on the fly
    for (int i = 0; i < depth; i++) {
      // lambda_ = x DOT currNode.w1 (project x onto the current node's INPUT
      // basis vector)
      F *w1s = _in_projection + current_node * w1s_sz;
      F *w2s = _out_projection + current_node * w2s_sz;

      F lambda_ = 0;
      for (int j = 0; j < w1s_sz; j++) {
        lambda_ += x[j] * w1s[j];
      }

      *y += lambda_ * w2s[col_index];

      // figure out index of node in next layer to visit
      current_node = (current_node * 2) + 1 + (lambda_ > 0);
    }
  }
  __syncthreads();
}

void fff_cuda_forward(int B, int w1s_sz, int w2s_sz, bf16 *y, bf16 *x,
                      bf16 *in_projection, bf16 *out_projection,
                      const unsigned int depth) {
  // NOTE: we have 32 threads per block in each dimension
  const int threads_x = 32;
  const int threads_y = 32;
  // compute the number of blocks in each dimension
  const int blocks_x =
      (w2s_sz + threads_x - 1) / threads_x;              // round up the blocks
  const int blocks_y = (B + threads_y - 1) / threads_y;  // round up the blocks

  // use a dim3 struct to specify the grid and block dimensions
  dim3 grid(blocks_x, blocks_y);
  dim3 block(threads_x, threads_y);

  kernel_forward<<<grid, block>>>(B, w1s_sz, w2s_sz, y, x, in_projection,
                                  out_projection, depth);

  cudaError_t err;
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  cudaError_t cudaStatus;
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "addKernel!\n",
            cudaStatus);
  }
}