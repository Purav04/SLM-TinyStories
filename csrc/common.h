#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32 (Phase 1 kernels are fp32-only)")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

// Row-wise kernels (softmax, layernorm) use one block per row, with a
// power-of-two thread count so the shared-memory tree reduction is exact.
inline int pick_row_block_size(int64_t N) {
  int block = 1;
  while (block < N && block < 1024) block <<= 1;
  return block;
}
