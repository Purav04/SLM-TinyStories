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

// Phase 2: shared-memory tiled GEMM. Each block cooperatively loads a
// TILE_DIM x TILE_DIM tile of A and B into shared memory once, then every
// thread in the block reuses those TILE_DIM values from shared memory
// instead of re-reading them from global memory — cutting global memory
// traffic by roughly a factor of TILE_DIM versus the naive kernel.
//
// Both tile loads are written so threadIdx.x always indexes whichever
// dimension is physically contiguous in memory, regardless of the
// TransA/TransB flags, so every global read is coalesced even when
// loading a "transposed" operand. The transpose is instead handled by
// how the compute loop indexes back into shared memory. The `+1` padding
// on each shared tile's second dimension avoids the bank conflicts that
// row-major transposed shared-memory access would otherwise cause.
#define TILE_DIM 32

template <bool TransA, bool TransB>
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
  __shared__ float As[TILE_DIM][TILE_DIM + 1];
  __shared__ float Bs[TILE_DIM][TILE_DIM + 1];

  int block_row = blockIdx.y * TILE_DIM;
  int block_col = blockIdx.x * TILE_DIM;
  int row = block_row + threadIdx.y;
  int col = block_col + threadIdx.x;

  float acc = 0.0f;
  int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

  for (int t = 0; t < num_tiles; ++t) {
    int k0 = t * TILE_DIM;

    if (!TransA) {
      // A is physically [M, K]; varying threadIdx.x walks contiguous K.
      int k = k0 + threadIdx.x;
      As[threadIdx.y][threadIdx.x] = (row < M && k < K) ? A[row * K + k] : 0.0f;
    } else {
      // A is physically [K, M]; varying threadIdx.x walks contiguous M.
      // Stored as As[k_local][m_local] so the compute loop below reads it
      // transposed.
      int k = k0 + threadIdx.y;
      int m = block_row + threadIdx.x;
      As[threadIdx.y][threadIdx.x] = (k < K && m < M) ? A[k * M + m] : 0.0f;
    }

    if (!TransB) {
      // B is physically [K, N]; varying threadIdx.x walks contiguous N.
      int k = k0 + threadIdx.y;
      Bs[threadIdx.y][threadIdx.x] = (k < K && col < N) ? B[k * N + col] : 0.0f;
    } else {
      // B is physically [N, K]; varying threadIdx.x walks contiguous K.
      // Stored as Bs[n_local][k_local] so the compute loop reads it
      // transposed.
      int k = k0 + threadIdx.x;
      int n = block_col + threadIdx.y;
      Bs[threadIdx.y][threadIdx.x] = (n < N && k < K) ? B[n * K + k] : 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < TILE_DIM; ++kk) {
      float a_val = TransA ? As[kk][threadIdx.y] : As[threadIdx.y][kk];
      float b_val = TransB ? Bs[threadIdx.x][kk] : Bs[kk][threadIdx.x];
      acc += a_val * b_val;
    }

    __syncthreads();
  }

  if (row < M && col < N) C[row * N + col] = acc;
}

static void launch_matmul_tiled(const float* A, const float* B, float* C,
                                 int M, int N, int K, bool transA, bool transB,
                                 cudaStream_t stream) {
  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

  if (!transA && !transB)
    matmul_tiled_kernel<false, false><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else if (!transA && transB)
    matmul_tiled_kernel<false, true><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else if (transA && !transB)
    matmul_tiled_kernel<true, false><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else
    matmul_tiled_kernel<true, true><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

torch::Tensor matmul_tiled_cuda(torch::Tensor A, torch::Tensor B, bool transA, bool transB) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul_tiled_cuda: expected 2D tensors");

  int M = transA ? A.size(1) : A.size(0);
  int K = transA ? A.size(0) : A.size(1);
  int K2 = transB ? B.size(1) : B.size(0);
  int N = transB ? B.size(0) : B.size(1);
  TORCH_CHECK(K == K2, "matmul_tiled_cuda: inner dimensions must match, got K=", K, " and K2=", K2);

  auto C = torch::empty({M, N}, A.options());
  launch_matmul_tiled(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                       M, N, K, transA, transB, at::cuda::getCurrentCUDAStream());
  return C;
}
