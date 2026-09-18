#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32 (Phase 1 kernels are fp32-only)")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

// Row-wise kernels (softmax, layernorm) use one block per row. Always a
// multiple of 32: blockReduceSum/Max below assume every warp is full, and
// __shfl_down_sync with a 0xffffffff mask is undefined if some lanes in
// the warp don't correspond to a real thread.
inline int pick_row_block_size(int64_t N) {
  int block = 32;
  while (block < N && block < 1024) block <<= 1;
  return block;
}

// Warp- and block-level sum/max reductions via __shfl_down_sync, replacing
// the naive shared-memory tree reduction (Phase 1). Butterfly-reduces
// within each warp using register shuffles (no shared memory, no
// __syncthreads needed for that part), then combines the per-warp results
// through a small shared array. Callers must have every thread in the
// block participate (no divergent early-outs before calling these).
__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
  __shared__ float warp_results[32];  // one slot per warp, supports up to 1024 threads/block
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);
  if (lane == 0) warp_results[wid] = val;
  __syncthreads();

  int num_warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < num_warps) ? warp_results[lane] : 0.0f;
  if (wid == 0) val = warpReduceSum(val);
  if (threadIdx.x == 0) warp_results[0] = val;
  __syncthreads();
  return warp_results[0];  // broadcast: every thread reads the same final value
}

__device__ __forceinline__ float blockReduceMax(float val) {
  __shared__ float warp_results[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val);
  if (lane == 0) warp_results[wid] = val;
  __syncthreads();

  int num_warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < num_warps) ? warp_results[lane] : -INFINITY;
  if (wid == 0) val = warpReduceMax(val);
  if (threadIdx.x == 0) warp_results[0] = val;
  __syncthreads();
  return warp_results[0];
}
