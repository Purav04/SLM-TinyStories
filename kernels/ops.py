"""Autograd-aware wrappers around the custom CUDA kernels, shaped as
drop-in replacements for nn.Linear / F.layer_norm / F.softmax.

Import this module only where CUDA kernels are actually used (model.py
does it lazily) — the extension must be built first with:

    pip install -e .
"""
import torch
import torch.nn as nn

import slm_cuda_kernels as _C


def _flatten(x):
    """Collapse all leading dims into one row dim; kernels only see 2D."""
    orig_shape = x.shape
    return x.reshape(-1, orig_shape[-1]).contiguous(), orig_shape


class CudaLinearFunction(torch.autograd.Function):
    # Phase 2: routed through the shared-memory tiled GEMM (_C.matmul_tiled)
    # instead of Phase 1's naive kernel (_C.matmul, still exposed for the
    # naive-vs-tiled-vs-torch benchmark comparison in benchmarks/bench.py).
    @staticmethod
    def forward(ctx, x, weight, bias):
        x2d, orig_shape = _flatten(x)
        out2d = _C.matmul_tiled(x2d, weight, False, True)  # x @ W^T
        if bias is not None:
            out2d = out2d + bias
        ctx.save_for_backward(x2d, weight, bias if bias is not None else torch.empty(0))
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        return out2d.reshape(*orig_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, _ = ctx.saved_tensors
        grad_out2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()

        grad_x = _C.matmul_tiled(grad_out2d, weight, False, False)   # grad_out @ W
        grad_weight = _C.matmul_tiled(grad_out2d, x2d, True, False)  # grad_out^T @ x
        grad_bias = grad_out2d.sum(dim=0) if ctx.has_bias else None

        return grad_x.reshape(ctx.orig_shape), grad_weight, grad_bias


class CudaLinear(nn.Module):
    """Drop-in replacement for nn.Linear backed by the custom CUDA GEMM kernel."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return CudaLinearFunction.apply(x, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class CudaSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x2d, orig_shape = _flatten(x)
        y2d = _C.softmax_forward(x2d)
        ctx.save_for_backward(y2d)
        ctx.orig_shape = orig_shape
        return y2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        (y2d,) = ctx.saved_tensors
        grad_out2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_x2d = _C.softmax_backward(grad_out2d, y2d)
        return grad_x2d.reshape(ctx.orig_shape)


def cuda_softmax(x, dim=-1):
    """Drop-in replacement for F.softmax(x, dim=-1). Only the last dim is supported."""
    assert dim == -1 or dim == x.dim() - 1, "cuda_softmax only supports softmax over the last dim"
    return CudaSoftmaxFunction.apply(x)


class CudaLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        x2d, orig_shape = _flatten(x)
        y2d, mean, rstd = _C.layernorm_forward(x2d, weight, bias, eps)
        ctx.save_for_backward(x2d, weight, mean, rstd)
        ctx.orig_shape = orig_shape
        return y2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, mean, rstd = ctx.saved_tensors
        grad_out2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_x2d, grad_weight, grad_bias = _C.layernorm_backward(
            grad_out2d, x2d, weight, mean, rstd
        )
        return grad_x2d.reshape(ctx.orig_shape), grad_weight, grad_bias, None


class CudaLayerNorm(nn.Module):
    """Drop-in replacement for the notebook's LayerNorm module (F.layer_norm-backed)."""

    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        bias = self.bias if self.bias is not None else torch.zeros_like(self.weight)
        return CudaLayerNormFunction.apply(x, self.weight, bias, self.eps)
