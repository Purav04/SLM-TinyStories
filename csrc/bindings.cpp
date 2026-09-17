#include <torch/extension.h>
#include <vector>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, bool transA, bool transB);

torch::Tensor softmax_forward_cuda(torch::Tensor x);
torch::Tensor softmax_backward_cuda(torch::Tensor grad_out, torch::Tensor y);

std::vector<torch::Tensor> layernorm_forward_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
std::vector<torch::Tensor> layernorm_backward_cuda(
    torch::Tensor grad_out, torch::Tensor x, torch::Tensor weight,
    torch::Tensor mean, torch::Tensor rstd);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul_cuda,
        "Naive CUDA matmul with optional operand transpose (CUDA)");
  m.def("softmax_forward", &softmax_forward_cuda, "Row-wise softmax forward (CUDA)");
  m.def("softmax_backward", &softmax_backward_cuda, "Row-wise softmax backward (CUDA)");
  m.def("layernorm_forward", &layernorm_forward_cuda, "LayerNorm forward (CUDA)");
  m.def("layernorm_backward", &layernorm_backward_cuda, "LayerNorm backward (CUDA)");
}
