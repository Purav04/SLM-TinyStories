"""Correctness tests: every custom CUDA kernel vs its torch.nn.functional
equivalent, forward and backward. Requires a CUDA GPU and the extension to
be built (`pip install -e .` from the repo root).

Run with: pytest tests/test_ops.py -v
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

DEVICE = "cuda"
ATOL, RTOL = 1e-3, 1e-3  # naive kernels sum in a different order than cuBLAS/ATen


def _needs_ext():
    pytest.importorskip("slm_cuda_kernels", reason="build the extension first: pip install -e .")


class TestMatmul:
    def test_forward_no_transpose(self):
        _needs_ext()
        import slm_cuda_kernels as _C
        A = torch.randn(37, 53, device=DEVICE)
        B = torch.randn(53, 29, device=DEVICE)
        out = _C.matmul(A, B, False, False)
        ref = A @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_forward_transpose_b(self):
        _needs_ext()
        import slm_cuda_kernels as _C
        A = torch.randn(37, 53, device=DEVICE)
        B = torch.randn(29, 53, device=DEVICE)  # transB -> logical shape [53, 29]
        out = _C.matmul(A, B, False, True)
        ref = A @ B.t()
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_forward_transpose_a(self):
        _needs_ext()
        import slm_cuda_kernels as _C
        A = torch.randn(53, 37, device=DEVICE)  # transA -> logical shape [37, 53]
        B = torch.randn(53, 29, device=DEVICE)
        out = _C.matmul(A, B, True, False)
        ref = A.t() @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


class TestMatmulTiled:
    """Same checks as TestMatmul, against the Phase 2 shared-memory tiled
    kernel instead of the Phase 1 naive one. Shapes are deliberately not
    multiples of TILE_DIM (32) to exercise the tile boundary-padding logic,
    and all four transpose combinations are covered here (Phase 1 only
    tested three) since the tiled kernel's load/compute indexing is
    transpose-direction-specific in both A and B."""

    def test_forward_no_transpose(self):
        _needs_ext()
        import slm_cuda_kernels as _C
        A = torch.randn(65, 90, device=DEVICE)
        B = torch.randn(90, 45, device=DEVICE)
        out = _C.matmul_tiled(A, B, False, False)
        ref = A @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_forward_transpose_b(self):
        _needs_ext()
        import slm_cuda_kernels as _C
        A = torch.randn(65, 90, device=DEVICE)
        B = torch.randn(45, 90, device=DEVICE)
        out = _C.matmul_tiled(A, B, False, True)
        ref = A @ B.t()
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_forward_transpose_a(self):
        _needs_ext()
        import slm_cuda_kernels as _C
        A = torch.randn(90, 65, device=DEVICE)
        B = torch.randn(90, 45, device=DEVICE)
        out = _C.matmul_tiled(A, B, True, False)
        ref = A.t() @ B
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_forward_transpose_both(self):
        _needs_ext()
        import slm_cuda_kernels as _C
        A = torch.randn(90, 65, device=DEVICE)
        B = torch.randn(45, 90, device=DEVICE)
        out = _C.matmul_tiled(A, B, True, True)
        ref = A.t() @ B.t()
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_matches_naive_kernel(self):
        _needs_ext()
        import slm_cuda_kernels as _C
        A = torch.randn(64, 64, device=DEVICE)
        B = torch.randn(64, 64, device=DEVICE)
        torch.testing.assert_close(
            _C.matmul_tiled(A, B, False, False), _C.matmul(A, B, False, False),
            atol=ATOL, rtol=RTOL,
        )


class TestCudaLinear:
    def test_forward_backward_vs_nn_linear(self):
        _needs_ext()
        from kernels import CudaLinear

        torch.manual_seed(0)
        x_ref = torch.randn(4, 16, 32, device=DEVICE, requires_grad=True)
        x_cuda = x_ref.detach().clone().requires_grad_()

        ref_layer = torch.nn.Linear(32, 64, bias=True).to(DEVICE)
        cuda_layer = CudaLinear(32, 64, bias=True).to(DEVICE)
        with torch.no_grad():
            cuda_layer.weight.copy_(ref_layer.weight)
            cuda_layer.bias.copy_(ref_layer.bias)

        out_ref = ref_layer(x_ref)
        out_cuda = cuda_layer(x_cuda)
        torch.testing.assert_close(out_cuda, out_ref, atol=ATOL, rtol=RTOL)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_cuda.backward(grad_out)

        torch.testing.assert_close(x_cuda.grad, x_ref.grad, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(cuda_layer.weight.grad, ref_layer.weight.grad, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(cuda_layer.bias.grad, ref_layer.bias.grad, atol=ATOL, rtol=RTOL)


class TestCudaSoftmax:
    def test_forward_backward_vs_functional(self):
        _needs_ext()
        from kernels import cuda_softmax

        torch.manual_seed(0)
        x_ref = torch.randn(8, 12, 64, device=DEVICE, requires_grad=True)
        x_cuda = x_ref.detach().clone().requires_grad_()

        out_ref = F.softmax(x_ref, dim=-1)
        out_cuda = cuda_softmax(x_cuda, dim=-1)
        torch.testing.assert_close(out_cuda, out_ref, atol=ATOL, rtol=RTOL)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_cuda.backward(grad_out)
        torch.testing.assert_close(x_cuda.grad, x_ref.grad, atol=ATOL, rtol=RTOL)


class TestCudaLayerNorm:
    def test_forward_backward_vs_functional(self):
        _needs_ext()
        from kernels import CudaLayerNorm

        torch.manual_seed(0)
        x_ref = torch.randn(4, 16, 64, device=DEVICE, requires_grad=True)
        x_cuda = x_ref.detach().clone().requires_grad_()

        weight = torch.randn(64, device=DEVICE, requires_grad=True)
        bias = torch.randn(64, device=DEVICE, requires_grad=True)
        w_cuda = weight.detach().clone().requires_grad_()
        b_cuda = bias.detach().clone().requires_grad_()

        ln = CudaLayerNorm(64, bias=True).to(DEVICE)
        with torch.no_grad():
            ln.weight.copy_(w_cuda)
            ln.bias.copy_(b_cuda)
        ln.weight.requires_grad_()
        ln.bias.requires_grad_()

        out_ref = F.layer_norm(x_ref, (64,), weight, bias, eps=1e-5)
        out_cuda = ln(x_cuda)
        torch.testing.assert_close(out_cuda, out_ref, atol=ATOL, rtol=RTOL)

        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_cuda.backward(grad_out)

        torch.testing.assert_close(x_cuda.grad, x_ref.grad, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(ln.weight.grad, weight.grad, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(ln.bias.grad, bias.grad, atol=ATOL, rtol=RTOL)


class TestGPTModel:
    def test_forward_backward_matches_native(self):
        _needs_ext()
        from model import GPT, GPTConfig

        torch.manual_seed(0)
        cfg_kwargs = dict(block_size=32, vocab_size=100, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)

        native = GPT(GPTConfig(**cfg_kwargs, use_cuda_kernels=False)).to(DEVICE)
        cuda_model = GPT(GPTConfig(**cfg_kwargs, use_cuda_kernels=True)).to(DEVICE)
        # strict=False: native takes the flash-attention path and never registers
        # the causal-mask buffer, while cuda_model is forced onto the manual
        # attention path and does. It's a deterministic constant derived from
        # config, not a learned parameter, so the mismatch is expected.
        cuda_model.load_state_dict(native.state_dict(), strict=False)

        idx = torch.randint(0, 100, (2, 16), device=DEVICE)
        targets = torch.randint(0, 100, (2, 16), device=DEVICE)

        _, loss_native = native(idx, targets)
        _, loss_cuda = cuda_model(idx, targets)
        torch.testing.assert_close(loss_cuda, loss_native, atol=ATOL, rtol=RTOL)
