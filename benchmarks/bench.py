"""Benchmark: naive vs tiled CUDA kernels vs native PyTorch (cuBLAS/ATen).

Extend later with Nsight Compute/Systems runs and a roofline analysis
(Phase 3), and add the fused-kernel variants once they land.

Run on a CUDA machine (e.g. ASU Sol): python benchmarks/bench.py
"""
import os
import sys
import torch
import torch.nn.functional as F

# `python benchmarks/bench.py` puts this file's own directory on sys.path,
# not the repo root — add the root explicitly so `kernels`/`model` (which
# live there, not as installed packages) are importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slm_cuda_kernels as _C
from kernels import CudaLinear, CudaLayerNorm, cuda_softmax

DEVICE = "cuda"
WARMUP, ITERS = 10, 100


def _time(fn, *args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def bench_matmul_variants(M=1024, N=1024, K=1024):
    """Raw GEMM, no autograd/bias overhead: naive vs tiled vs cuBLAS. This is
    the naive-vs-tiled-vs-cuBLAS number for the Phase 3 speedup chart.
    Kept at 1024^3 (not e.g. 4096^3) on purpose — the naive O(N^3),
    no-reuse kernel is slow enough that a larger size makes this take
    minutes just from the naive leg."""
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)

    t_naive = _time(lambda: _C.matmul(A, B, False, False), warmup=3, iters=10)
    t_tiled = _time(lambda: _C.matmul_tiled(A, B, False, False))
    t_torch = _time(lambda: A @ B)

    flops = 2 * M * N * K  # multiply-add = 2 FLOPs
    gflops_per_sec = lambda t_ms: flops * 1e-6 / t_ms  # flops * 1e-9 / (t_ms * 1e-3)
    print(f"Matmul  [{M}x{K} @ {K}x{N}]")
    print(f"  naive : {t_naive:.3f}ms  ({gflops_per_sec(t_naive):.1f} GFLOP/s)")
    print(f"  tiled : {t_tiled:.3f}ms  ({gflops_per_sec(t_tiled):.1f} GFLOP/s)  speedup over naive={t_naive/t_tiled:.2f}x")
    print(f"  torch : {t_torch:.3f}ms  ({gflops_per_sec(t_torch):.1f} GFLOP/s)  tiled is {t_torch/t_tiled*100:.1f}% of cuBLAS throughput")


def bench_linear(batch=32, seq=256, d_in=768, d_out=768):
    x = torch.randn(batch, seq, d_in, device=DEVICE)
    grad_out = torch.randn(batch, seq, d_out, device=DEVICE)

    ref = torch.nn.Linear(d_in, d_out).to(DEVICE)
    cuda_layer = CudaLinear(d_in, d_out).to(DEVICE)
    cuda_layer.weight.data.copy_(ref.weight.data)
    cuda_layer.bias.data.copy_(ref.bias.data)

    def fwd_bwd(layer, x):
        x = x.detach().requires_grad_()
        out = layer(x)
        out.backward(grad_out)
        return out

    t_ref = _time(fwd_bwd, ref, x)
    t_cuda = _time(fwd_bwd, cuda_layer, x)
    print(f"Linear  [{batch}x{seq}x{d_in}->{d_out}]  torch={t_ref:.3f}ms  cuda={t_cuda:.3f}ms  speedup(torch/cuda)={t_ref/t_cuda:.2f}x")


def bench_softmax(batch=32, heads=12, seq=256):
    x = torch.randn(batch, heads, seq, seq, device=DEVICE)
    grad_out = torch.randn_like(x)

    def fwd_bwd_ref(x):
        x = x.detach().requires_grad_()
        out = F.softmax(x, dim=-1)
        out.backward(grad_out)
        return out

    def fwd_bwd_cuda(x):
        x = x.detach().requires_grad_()
        out = cuda_softmax(x, dim=-1)
        out.backward(grad_out)
        return out

    t_ref = _time(fwd_bwd_ref, x)
    t_cuda = _time(fwd_bwd_cuda, x)
    print(f"Softmax [{batch}x{heads}x{seq}x{seq}]  torch={t_ref:.3f}ms  cuda={t_cuda:.3f}ms  speedup(torch/cuda)={t_ref/t_cuda:.2f}x")


def bench_layernorm(batch=32, seq=256, d=768):
    x = torch.randn(batch, seq, d, device=DEVICE)
    grad_out = torch.randn_like(x)
    weight = torch.randn(d, device=DEVICE)
    bias = torch.randn(d, device=DEVICE)

    cuda_ln = CudaLayerNorm(d).to(DEVICE)
    cuda_ln.weight.data.copy_(weight)
    cuda_ln.bias.data.copy_(bias)

    def fwd_bwd_ref(x):
        x = x.detach().requires_grad_()
        out = F.layer_norm(x, (d,), weight, bias, eps=1e-5)
        out.backward(grad_out)
        return out

    def fwd_bwd_cuda(x):
        x = x.detach().requires_grad_()
        out = cuda_ln(x)
        out.backward(grad_out)
        return out

    t_ref = _time(fwd_bwd_ref, x)
    t_cuda = _time(fwd_bwd_cuda, x)
    print(f"LayerNorm [{batch}x{seq}x{d}]  torch={t_ref:.3f}ms  cuda={t_cuda:.3f}ms  speedup(torch/cuda)={t_ref/t_cuda:.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "benchmarks need a CUDA GPU"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    bench_matmul_variants()
    bench_linear()
    bench_softmax()
    bench_layernorm()
