"""Phase 1/3 benchmark skeleton: naive CUDA kernels vs native PyTorch.

This gives wall-clock numbers now; extend it later with Nsight
Compute/Systems runs and a roofline analysis (Phase 3), and add the
tiled/fused kernel variants as new rows once Phase 2 lands.

Run on a CUDA machine (e.g. ASU Sol): python benchmarks/bench.py
"""
import torch
import torch.nn.functional as F

from kernels import CudaLinear, CudaLayerNorm, cuda_softmax

DEVICE = "cuda"
WARMUP, ITERS = 10, 100


def _time(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS  # ms/iter


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
    bench_linear()
    bench_softmax()
    bench_layernorm()
