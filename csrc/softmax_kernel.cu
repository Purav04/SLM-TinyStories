// Row-wise softmax, one CUDA block per row. Naive block-level tree
// reduction in shared memory for the max and sum passes (Phase 2 swaps
// this for __shfl_sync warp reductions). Handles the numerically-stable
// max-subtraction form since this feeds attention scores that can be
// large in magnitude.
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

__global__ void softmax_forward_kernel(
    const float* __restrict__ x, float* __restrict__ y, int N) {
  extern __shared__ float shared[];
  int row = blockIdx.x;
  const float* x_row = x + (size_t)row * N;
  float* y_row = y + (size_t)row * N;

  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    local_max = fmaxf(local_max, x_row[i]);
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  float row_max = shared[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float e = expf(x_row[i] - row_max);
    y_row[i] = e;  // stash exp(x - max), normalize in the second pass
    local_sum += e;
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  float row_sum = shared[0];
  __syncthreads();

  for (int i = threadIdx.x; i < N; i += blockDim.x)
    y_row[i] = y_row[i] / row_sum;
}

// dx = y * (dy - sum(dy * y)), the standard softmax vector-Jacobian product.
__global__ void softmax_backward_kernel(
    const float* __restrict__ dy, const float* __restrict__ y,
    float* __restrict__ dx, int N) {
  extern __shared__ float shared[];
  int row = blockIdx.x;
  const float* dy_row = dy + (size_t)row * N;
  const float* y_row = y + (size_t)row * N;
  float* dx_row = dx + (size_t)row * N;

  float local = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    local += dy_row[i] * y_row[i];
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  float dot = shared[0];
  __syncthreads();

  for (int i = threadIdx.x; i < N; i += blockDim.x)
    dx_row[i] = y_row[i] * (dy_row[i] - dot);
}

torch::Tensor softmax_forward_cuda(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() == 2, "softmax_forward_cuda: expected a 2D [rows, N] tensor");

  int64_t M = x.size(0), N = x.size(1);
  auto y = torch::empty_like(x);
  int block = pick_row_block_size(N);
  size_t shmem = block * sizeof(float);

  softmax_forward_kernel<<<M, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
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
  size_t shmem = block * sizeof(float);

  softmax_backward_kernel<<<M, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
      grad_out.data_ptr<float>(), y.data_ptr<float>(), dx.data_ptr<float>(), N);
  return dx;
}
