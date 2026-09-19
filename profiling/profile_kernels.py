"""Profiling harness for Nsight Compute / Nsight Systems (Phase 3).

Each kernel call is bracketed with an NVTX range so Nsight Systems' timeline
shows clearly labeled regions, and both tools can target a specific call
instead of profiling the whole process. Two matmul sizes are profiled on
purpose:

  - "model": 8192x768 @ 768x768, matching the shapes actually used by
    CudaLinear in the real GPT model (batch*seq=8192, n_embd=768). Small
    enough that all three ~4MB-ish matrices fit comfortably in the A100's
    40MB L2 cache, which is why the naive kernel's earlier benchmark
    numbers looked better than a truly memory-bound kernel "should".
  - "large": 4096x4096 @ 4096x4096, ~192MB total working set, well past
    L2. This is the size that actually exposes the naive kernel's true
    memory-bound behavior for the roofline analysis, uncomplicated by
    cache effects.

Run directly first (no profiler) to sanity check it runs end to end:
    python profiling/profile_kernels.py
Then wrap it with ncu/nsys per README.md's Phase 3 instructions.
"""
import os
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import slm_cuda_kernels as _C
from kernels import CudaLayerNorm, cuda_softmax

DEVICE = "cuda"


@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def profile_matmul(M, N, K, tag):
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    torch.cuda.synchronize()

    with nvtx_range(f"matmul_naive_{tag}"):
        _C.matmul(A, B, False, False)
    torch.cuda.synchronize()

    with nvtx_range(f"matmul_tiled_{tag}"):
        _C.matmul_tiled(A, B, False, False)
    torch.cuda.synchronize()

    with nvtx_range(f"matmul_torch_{tag}"):
        A @ B
    torch.cuda.synchronize()


def profile_softmax(batch, heads, seq):
    x = torch.randn(batch, heads, seq, seq, device=DEVICE)
    torch.cuda.synchronize()

    with nvtx_range("softmax_cuda"):
        cuda_softmax(x, dim=-1)
    torch.cuda.synchronize()

    with nvtx_range("softmax_torch"):
        F.softmax(x, dim=-1)
    torch.cuda.synchronize()


def profile_layernorm(batch, seq, d):
    x = torch.randn(batch, seq, d, device=DEVICE)
    ln = CudaLayerNorm(d).to(DEVICE)
    torch.cuda.synchronize()

    with nvtx_range("layernorm_cuda"):
        ln(x)
    torch.cuda.synchronize()

    with nvtx_range("layernorm_torch"):
        F.layer_norm(x, (d,), ln.weight, ln.bias, eps=1e-5)
    torch.cuda.synchronize()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "profiling needs a CUDA GPU"
    profile_matmul(8192, 768, 768, tag="model")
    profile_matmul(4096, 4096, 4096, tag="large")
    profile_softmax(32, 12, 256)
    profile_layernorm(32, 256, 768)
    print("done")
