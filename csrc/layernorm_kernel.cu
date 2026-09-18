// LayerNorm, one CUDA block per row. Forward saves per-row mean and rstd
// (1/sqrt(var+eps)) so backward doesn't recompute them.
//
// Phase 2: the per-row reductions (mean, variance, and backward's
// sum(dxhat)/sum(dxhat*xhat)) use __shfl_down_sync warp shuffles
// (blockReduceSum in common.h) instead of Phase 1's shared-memory tree.
// dweight/dbias backward still uses atomicAdd across rows — that's a
// separate, still-open optimization (a proper column-reduction kernel),
// not part of this pass.
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"

__global__ void layernorm_forward_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out,
    int N, float eps) {
  int row = blockIdx.x;
  const float* x_row = x + (size_t)row * N;
  float* y_row = y + (size_t)row * N;

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) local_sum += x_row[i];
  float mean = blockReduceSum(local_sum) / N;

  float local_var = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float d = x_row[i] - mean;
    local_var += d * d;
  }
  float var = blockReduceSum(local_var) / N;
  float rstd = rsqrtf(var + eps);

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
  int row = blockIdx.x;
  const float* x_row = x + (size_t)row * N;
  const float* dy_row = dy + (size_t)row * N;
  float* dx_row = dx + (size_t)row * N;
  float row_mean = mean[row];
  float row_rstd = rstd[row];

  float local_a = 0.0f;  // sum(dxhat)
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float xhat = (x_row[i] - row_mean) * row_rstd;
    float dxhat = dy_row[i] * weight[i];
    local_a += dxhat;
    atomicAdd(&dweight[i], dy_row[i] * xhat);
    atomicAdd(&dbias[i], dy_row[i]);
  }
  float mean_dxhat = blockReduceSum(local_a) / N;

  float local_b = 0.0f;  // sum(dxhat * xhat)
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float xhat = (x_row[i] - row_mean) * row_rstd;
    float dxhat = dy_row[i] * weight[i];
    local_b += dxhat * xhat;
  }
  float mean_dxhat_xhat = blockReduceSum(local_b) / N;

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
  layernorm_forward_kernel<<<M, block, 0, at::cuda::getCurrentCUDAStream()>>>(
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
  layernorm_backward_kernel<<<M, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      grad_out.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), dx.data_ptr<float>(),
      dweight.data_ptr<float>(), dbias.data_ptr<float>(), N);

  return {dx, dweight, dbias};
}
