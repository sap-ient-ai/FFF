#include <string.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

// CUDA forward declarations

torch::Tensor fff_cuda_forward(torch::Tensor x, torch::Tensor in_projection,
                               torch::Tensor out_projection,
                               const unsigned int depth);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) \
  AT_ASSERTM(x.device().type() == torch::kCUDA, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IDENTITY(x) AT_ASSERTM(x == 1, #x " must be 1")
#define CHECK_POSITIVE(x) AT_ASSERTM(x > 0, #x " must be positive")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor fff_forward(torch::Tensor x, torch::Tensor in_projection,
                          torch::Tensor out_projection, unsigned int depth) {
  CHECK_INPUT(x);
  CHECK_INPUT(in_projection);
  CHECK_INPUT(out_projection);

  return fff_cuda_forward(x, in_projection, out_projection, depth);
}

std::vector<torch::Tensor> fff_backward(torch::Tensor inputs) {
  CHECK_INPUT(inputs);
  std::cout << "fff_backward not implemented" << std::endl;
  return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fff_forward, "FFF forward (CUDA)");
  m.def("backward", &fff_backward, "FFF backward (CUDA)");
}
