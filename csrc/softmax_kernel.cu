// Row-wise softmax, one CUDA block per row.
//
// Phase 2: reductions use __shfl_down_sync warp shuffles (blockReduceMax/
// Sum in common.h) instead of Phase 1's shared-memory tree — most of the
// reduction happens in registers within a warp, with only one value per
// warp ever touching shared memory. Numerically stable (max-subtracted)
// since this feeds attention scores that can be large in magnitude.
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

__global__ void softmax_forward_kernel(
    const float* __restrict__ x, float* __restrict__ y, int N) {
  int row = blockIdx.x;
  const float* x_row = x + (size_t)row * N;
  float* y_row = y + (size_t)row * N;

  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    local_max = fmaxf(local_max, x_row[i]);
  float row_max = blockReduceMax(local_max);

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float e = expf(x_row[i] - row_max);
    y_row[i] = e;  // stash exp(x - max), normalize in the second pass
    local_sum += e;
  }
  float row_sum = blockReduceSum(local_sum);

  for (int i = threadIdx.x; i < N; i += blockDim.x)
    y_row[i] = y_row[i] / row_sum;
}

// dx = y * (dy - sum(dy * y)), the standard softmax vector-Jacobian product.
__global__ void softmax_backward_kernel(
    const float* __restrict__ dy, const float* __restrict__ y,
    float* __restrict__ dx, int N) {
  int row = blockIdx.x;
  const float* dy_row = dy + (size_t)row * N;
  const float* y_row = y + (size_t)row * N;
  float* dx_row = dx + (size_t)row * N;

  float local = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    local += dy_row[i] * y_row[i];
  float dot = blockReduceSum(local);

  for (int i = threadIdx.x; i < N; i += blockDim.x)
    dx_row[i] = y_row[i] * (dy_row[i] - dot);
}

torch::Tensor softmax_forward_cuda(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() == 2, "softmax_forward_cuda: expected a 2D [rows, N] tensor");

  int64_t M = x.size(0), N = x.size(1);
  auto y = torch::empty_like(x);
  int block = pick_row_block_size(N);

  softmax_forward_kernel<<<M, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), N);
  return y;
}

torch::Tensor softmax_backward_cuda(torch::Tensor grad_out, torch::Tensor y) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(y);
  TORCH_CHECK(grad_out.sizes() == y.sizes(), "softmax_backward_cuda: shape mismatch");

  int64_t M = y.size(0), N = y.size(1);
  auto dx = torch::empty_like(y);
  int block = pick_row_block_size(N);

  softmax_backward_kernel<<<M, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      grad_out.data_ptr<float>(), y.data_ptr<float>(), dx.data_ptr<float>(), N);
  return dx;
}
