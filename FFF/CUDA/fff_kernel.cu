#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef at::FloatType fp32;

template <typename F>
__global__ void kernel_forward(const int B, const int w1s_sz, const int w2s_sz,
                               F *__restrict__ const _y,
                               F *__restrict__ const _x,
                               F *__restrict__ const _in_projection,
                               F *__restrict__ const _out_projection,
                               const unsigned int depth) {
  // get row index
  const int row_index = blockIdx.x * blockDim.x + threadIdx.x;

  // check if the row and column indices are valid
  if (row_index < w1s_sz) {
    // initialize the current node to zero
    int current_node = 0;
    // initialize the output vector to zero
    // F *y = _y + row_index * w2s_sz;
    F *y = _y + row_index * w2s_sz;
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

      // calculate the output of the current node
      // y += lambda_ * currNode.w2 (project lambda_ onto the current node's
      // OUTPUT basis vector)
      for (int j = 0; j < w2s_sz; j++) {
        y[j] += lambda_ * w2s[j];
      }

      // figure out index of node in next layer to visit
      current_node = (current_node * 2) + 1 + (lambda_ > 0 ? 1 : 0);
    }
  }
  __syncthreads();
}

void fff_cuda_forward(int B, int w1s_sz, int w2s_sz, float *y, float *x,
                      float *in_projection, float *out_projection,
                      const unsigned int depth) {
  // NOTE: we have 32 threads per block in each dimension
  const int n_threads = 32;
  const int n_blocks = (B + n_threads - 1) / n_threads;

  kernel_forward<<<n_blocks, n_threads>>>(B, w1s_sz, w2s_sz, y, x, in_projection,
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