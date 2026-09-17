// Naive CUDA GEMM: one thread per output element, no tiling, no shared
// memory. This is the Phase 1 "get it running" version — every thread
// re-reads its row of A and column of B from global memory K times, so
// it's heavily memory-bound. Phase 2 replaces this with a shared-memory
// tiled kernel.
//
// Computes C = op(A) @ op(B), where op(X) is X or X^T depending on the
// TransA/TransB flags. Supporting both transpose combinations here lets
// the same kernel serve the linear forward pass (x @ W^T) and both
// backward passes (grad_out @ W, grad_out^T @ x) without a separate
// transpose-then-matmul step.
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

template <bool TransA, bool TransB>
__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  // Non-transposed operand is [rows x K] row-major -> element (r,k) = X[r*K+k].
  // Transposed operand is stored as [K x rows] row-major, i.e. the tensor
  // we were actually handed has shape [K, rows] -> element (r,k) = X[k*rows+r].
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    float a = TransA ? A[k * M + row] : A[row * K + k];
    float b = TransB ? B[col * K + k] : B[k * N + col];
    acc += a * b;
  }
  C[row * N + col] = acc;
}

static void launch_matmul(const float* A, const float* B, float* C,
                           int M, int N, int K, bool transA, bool transB,
                           cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  if (!transA && !transB)
    matmul_naive_kernel<false, false><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else if (!transA && transB)
    matmul_naive_kernel<false, true><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else if (transA && !transB)
    matmul_naive_kernel<true, false><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else
    matmul_naive_kernel<true, true><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

// A, B are 2D tensors. transA/transB say whether to treat the tensor as
// pre-transposed (i.e. the tensor's own shape is the transpose of the
// logical operand shape) rather than transposing in Python.
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, bool transA, bool transB) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul_cuda: expected 2D tensors");

  int M = transA ? A.size(1) : A.size(0);
  int K = transA ? A.size(0) : A.size(1);
  int K2 = transB ? B.size(1) : B.size(0);
  int N = transB ? B.size(0) : B.size(1);
  TORCH_CHECK(K == K2, "matmul_cuda: inner dimensions must match, got K=", K, " and K2=", K2);

  auto C = torch::empty({M, N}, A.options());
  launch_matmul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                M, N, K, transA, transB, at::cuda::getCurrentCUDAStream());
  return C;
}
