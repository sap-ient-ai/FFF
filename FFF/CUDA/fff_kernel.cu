#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t z) {
  return z * (z > 0);
}

template <typename scalar_t>
__global__ void fff_cuda_forward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> x,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> in_projection,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> out_projection,
    const unsigned int depth,
    const unsigned int in_width,
    const unsigned int out_width
  ) {
  // compute which row of inputs we're dealing with
  const int row_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = x.size(1);

  // zero the output
  for (int i = 0; i < in_width; ++i) {
    output[row_index][i] = 0;
  }

  if (row_index < x.size(0)) {
    int current_node = 0;
    for (int current_depth = 0; current_depth <= depth; ++current_depth) {
        double acc = 0;

        #pragma unroll
        for (int i = 0; i < in_width;++i) {
            acc += x[row_index][i] * in_projection[current_node][i];
        }

        // compute the activation
        double activation = relu(acc);

        // compute the output contribution due to the current node
        #pragma unroll
        for (int i = 0; i < out_width; ++i) {
            output[row_index][i] += activation * out_projection[current_node][i];
        }

        // decide where to move to (left or right child)
        current_node = (current_node<<1) + 1 + (acc > 0);
    }
  }
}
} // namespace

torch::Tensor fff_cuda_forward(
	torch::Tensor x,
	torch::Tensor in_projection,
	torch::Tensor out_projection,
	const unsigned int depth
) {

  auto output = torch::empty(
    {x.size(0), out_projection.size(1)},
    torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(x.device())
  );

  const int batch_size = x.size(0);
  const int threads = 32;
  const int blocks = (batch_size + threads - 1) / threads;

  const int in_width = in_projection.size(1);
  const int out_width = out_projection.size(1);

  AT_DISPATCH_FLOATING_TYPES(in_projection.type(), "fff_forward_cuda", ([&] {
    fff_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        in_projection.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        out_projection.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        depth,
        in_width,
        out_width
    );
  }));

  cudaError_t err;
  err = cudaGetLastError();
  if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
  }

  cudaError_t cudaStatus;
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
  }

  return output;
}