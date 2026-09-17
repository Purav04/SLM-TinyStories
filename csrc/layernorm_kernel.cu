// LayerNorm, one CUDA block per row. Forward saves per-row mean and rstd
// (1/sqrt(var+eps)) so backward doesn't recompute them. Backward uses the
// standard LayerNorm gradient formula (see e.g. the "Deep Learning" LN
// derivation): two per-row reductions for dx, plus a column-wise
// reduction across rows for dweight/dbias done here via atomicAdd — the
// naive-but-correct choice for Phase 1. Phase 2 replaces the atomics with
// a proper reduction kernel.
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

__global__ void layernorm_forward_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out,
    int N, float eps) {
  extern __shared__ float shared[];
  int row = blockIdx.x;
  const float* x_row = x + (size_t)row * N;
  float* y_row = y + (size_t)row * N;

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) local_sum += x_row[i];
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  float mean = shared[0] / N;
  __syncthreads();

  float local_var = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float d = x_row[i] - mean;
    local_var += d * d;
  }
  shared[threadIdx.x] = local_var;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  float var = shared[0] / N;
  float rstd = rsqrtf(var + eps);
  __syncthreads();

  if (threadIdx.x == 0) {
    mean_out[row] = mean;
    rstd_out[row] = rstd;
  }

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float norm = (x_row[i] - mean) * rstd;
    y_row[i] = norm * weight[i] + bias[i];
  }
}

__global__ void layernorm_backward_kernel(
    const float* __restrict__ dy, const float* __restrict__ x,
    const float* __restrict__ weight, const float* __restrict__ mean,
    const float* __restrict__ rstd, float* __restrict__ dx,
    float* __restrict__ dweight, float* __restrict__ dbias, int N) {
  extern __shared__ float shared[];
  float* s_a = shared;              // sum(dxhat)
  float* s_b = shared + blockDim.x; // sum(dxhat * xhat)

  int row = blockIdx.x;
  const float* x_row = x + (size_t)row * N;
  const float* dy_row = dy + (size_t)row * N;
  float* dx_row = dx + (size_t)row * N;
  float row_mean = mean[row];
  float row_rstd = rstd[row];

  float local_a = 0.0f, local_b = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float xhat = (x_row[i] - row_mean) * row_rstd;
    float dxhat = dy_row[i] * weight[i];
    local_a += dxhat;
    local_b += dxhat * xhat;
    atomicAdd(&dweight[i], dy_row[i] * xhat);
    atomicAdd(&dbias[i], dy_row[i]);
  }
  s_a[threadIdx.x] = local_a;
  s_b[threadIdx.x] = local_b;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_a[threadIdx.x] += s_a[threadIdx.x + stride];
      s_b[threadIdx.x] += s_b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float mean_dxhat = s_a[0] / N;
  float mean_dxhat_xhat = s_b[0] / N;
  __syncthreads();

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float xhat = (x_row[i] - row_mean) * row_rstd;
    float dxhat = dy_row[i] * weight[i];
    dx_row[i] = row_rstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
  }
}

std::vector<torch::Tensor> layernorm_forward_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  TORCH_CHECK(x.dim() == 2, "layernorm_forward_cuda: expected a 2D [rows, N] tensor");

  int64_t M = x.size(0), N = x.size(1);
  auto y = torch::empty_like(x);
  auto mean = torch::empty({M}, x.options());
  auto rstd = torch::empty({M}, x.options());

  int block = pick_row_block_size(N);
  size_t shmem = block * sizeof(float);
  layernorm_forward_kernel<<<M, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
      y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
      N, static_cast<float>(eps));

  return {y, mean, rstd};
}

std::vector<torch::Tensor> layernorm_backward_cuda(
    torch::Tensor grad_out, torch::Tensor x, torch::Tensor weight,
    torch::Tensor mean, torch::Tensor rstd) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(mean);
  CHECK_INPUT(rstd);

  int64_t M = x.size(0), N = x.size(1);
  auto dx = torch::empty_like(x);
  auto dweight = torch::zeros_like(weight);
  auto dbias = torch::zeros_like(weight);

  int block = pick_row_block_size(N);
  size_t shmem = 2 * block * sizeof(float);
  layernorm_backward_kernel<<<M, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
      grad_out.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), dx.data_ptr<float>(),
      dweight.data_ptr<float>(), dbias.data_ptr<float>(), N);

  return {dx, dweight, dbias};
}
