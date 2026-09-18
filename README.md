# SLM-TinyStories

A from-scratch nanoGPT-style transformer trained on TinyStories, with a
custom CUDA kernel library replacing PyTorch's built-in matmul, softmax,
and layernorm ops.

## Project layout

- `SLM_TinyStories.ipynb`, `SLM - TinyStories.ipynb` — original training notebooks.
- `model.py` — the transformer (`GPT`, `Block`, `CausalSelfAttention`, `MLP`, `LayerNorm`), extracted from the notebooks. `GPTConfig(use_cuda_kernels=True)` routes every `nn.Linear`, the `LayerNorm`, and the non-flash attention softmax through the custom CUDA kernels instead of PyTorch's.
- `csrc/` — CUDA kernel sources (`matmul_kernel.cu`, `softmax_kernel.cu`, `layernorm_kernel.cu`) and the pybind11 bindings (`bindings.cpp`).
- `kernels/` — `torch.autograd.Function` wrappers (`CudaLinear`, `CudaLayerNorm`, `cuda_softmax`) that make the kernels drop-in, autograd-compatible replacements.
- `setup.py` — builds the `slm_cuda_kernels` extension via `torch.utils.cpp_extension.CUDAExtension`.
- `tests/test_ops.py` — correctness: every custom kernel vs its `torch.nn.functional` equivalent, forward and backward.
- `benchmarks/bench.py` — wall-clock timing: custom kernels vs native PyTorch.

## Status

**Phase 1 (naive kernels) — verified on Sol (A100).** `pytest tests/test_ops.py -v`
passed 7/7: matmul against all transpose combinations, `CudaLinear`/
`cuda_softmax`/`CudaLayerNorm` forward+backward against their
`torch.nn.functional` equivalents, and a full `GPT` forward+backward
matching between the native and CUDA-kernel code paths.

**Phase 2 (tiling, coalescing, warp shuffles) — implemented, not yet run
on Sol.** Like Phase 1's initial scaffold, this was written without a CUDA
toolchain available — build and test it before trusting it:

- `matmul_tiled` (`csrc/matmul_kernel.cu`): shared-memory tiled GEMM, `TILE_DIM=32`. Every global-memory tile load has `threadIdx.x` mapped to whichever dimension is physically contiguous — regardless of the logical transpose flag — so all four transpose combinations load coalesced; the transpose is instead handled by how the compute loop indexes back into shared memory. Shared tiles are padded (`[TILE][TILE+1]`) to avoid bank conflicts on the transposed reads. `CudaLinear` now uses this by default; the Phase 1 naive kernel (`matmul`) is kept for the benchmark comparison.
- Softmax and LayerNorm's row reductions (`csrc/common.h`: `blockReduceSum`/`blockReduceMax`) now use `__shfl_down_sync` warp-shuffle reduction instead of Phase 1's shared-memory tree — most of the reduction happens in registers within a warp, with only one value per warp touching shared memory.
- `benchmarks/bench.py` gained `bench_matmul_variants()`: naive vs tiled vs cuBLAS throughput (GFLOP/s) on a raw GEMM, no autograd overhead — the number for the eventual speedup chart.
- New tests: `TestMatmulTiled` covers all four transpose combinations (Phase 1 only covered three) at non-tile-multiple shapes, to exercise the boundary-padding logic, plus a direct naive-vs-tiled equality check.

Still open, deferred to a follow-up pass:

- LayerNorm's `dweight`/`dbias` backward still uses `atomicAdd` across rows instead of a proper column-reduction kernel.
- No fused kernels yet (bias+GELU, QK^T+softmax) — the attention QK^T matmul and the softmax over attention scores are still two separate ops.

## Building and testing on ASU Sol

Verified working on Sol with an A100 (`sm_80`), Python 3.13, torch 2.14.0,
CUDA 13.2, inside a venv named `slm`.

```bash
# 1. confirm your GPU generation (tiling/shared-memory sizing in Phase 2
#    depends on it — this project targets sm_80/Ampere)
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# 2. load a CUDA toolkit module (provides nvcc + sets CUDA_HOME)
module load cuda-13.2.1-gcc-13.2.0

# 3. load a matching GCC module too — this is required. The CUDA module's
#    name only describes what compiler *built* it; it does not put a
#    working compiler on your PATH. Without this step you're left with
#    Rocky Linux 8's default system GCC (8.5), which is too old for
#    torch's headers (torch 2.14 requires real C++20 support: concepts,
#    std::strong_ordering, string_view::starts_with/ends_with, etc.) and
#    fails with errors like "compare: No such file or directory" or
#    "unrecognized command line option '-std=c++20'".
module load gcc-13.2.0-gcc-12.1.0
which g++ && g++ --version   # must show >=10, ideally matching the CUDA module (13.2.0)

# 4. build the extension
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required: by default pip builds the package in
an isolated environment that doesn't have your already-installed `torch`
in it, and `setup.py` imports `torch.utils.cpp_extension` at import time —
without the flag you'll hit `ModuleNotFoundError: No module named 'torch'`
even though `torch` is installed and importable in your normal environment.

If you change compiler/CUDA modules later, do a clean rebuild rather than
an incremental one — stale object files from a different toolchain can
produce confusing errors:

```bash
rm -rf build *.egg-info
pip install -e . --no-build-isolation
```

Once it builds, run the actual test of correctness:

```bash
pytest tests/test_ops.py -v
python benchmarks/bench.py
```

## Next steps

- Verify Phase 2 on Sol: `pip install -e . --no-build-isolation` (clean rebuild), `pytest tests/test_ops.py -v`, `python benchmarks/bench.py`.
- Fused bias+GELU and QK^T+softmax kernels (Phase 2 fusion work, not yet started).
- A proper reduction kernel for LayerNorm's `dweight`/`dbias` instead of `atomicAdd`.
- Nsight Compute / Nsight Systems profiling on Sol; roofline analysis (compute- vs memory-bound, %peak FLOPs/bandwidth) — Phase 3.
- Naive → tiled → fused speedup chart vs native PyTorch and cuBLAS — Phase 3.
