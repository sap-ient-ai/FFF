#include <string.h>
#include <torch/extension.h>

#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half half;
typedef at::FloatType fp32;

#include <iostream>
#include <vector>

// CUDA forward declarations

void fff_cuda_forward(int B, int w1s_sz, int w2s_sz, float *y, float *x,
                      float *in_projection, float *out_projection,
                      const unsigned int depth);

void forward(int64_t B, int64_t w1s_sz, int64_t w2s_sz,
                      torch::Tensor &y, torch::Tensor &x,
                      torch::Tensor &in_projection,
                      torch::Tensor &out_projection, int64_t depth) {
  return fff_cuda_forward(B, w1s_sz, w2s_sz, y.data_ptr<float>(),
                          x.data_ptr<float>(), in_projection.data_ptr<float>(),
                          out_projection.data_ptr<float>(), depth);
}

std::vector<torch::Tensor> backward(torch::Tensor inputs) {
  std::cout << "fff_backward not implemented" << std::endl;
  return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "FFF forward (CUDA)");
  m.def("backward", &backward, "FFF backward (CUDA)");
}

TORCH_LIBRARY(fff_cuda_, m) {
  m.def("forward", forward);
  m.def("backward", backward);
}