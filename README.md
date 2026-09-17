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

## Status: Phase 1 (naive kernels) — verified on Sol (A100)

Builds and passes `pytest tests/test_ops.py -v` (7/7) on Sol: matmul
against all transpose combinations, `CudaLinear`/`cuda_softmax`/
`CudaLayerNorm` forward+backward against their `torch.nn.functional`
equivalents, and a full `GPT` forward+backward matching between the
native and CUDA-kernel code paths.

All three kernels are intentionally unoptimized — one thread per output
element for matmul, one block per row with a shared-memory tree reduction
for softmax/layernorm, no tiling, no warp shuffles, float32 only. The goal
here is a correct, autograd-wired baseline to optimize and benchmark
against. Known Phase 1 shortcuts, called out for Phase 2:

- LayerNorm's `dweight`/`dbias` backward uses `atomicAdd` across rows instead of a proper reduction kernel.
- Softmax/layernorm reductions are block-level shared-memory trees, not `__shfl_sync` warp reductions.
- Matmul has no shared-memory tiling, so it's heavily memory-bound.
- The attention QK^T matmul and the softmax over attention scores are still two separate ops — no flash-attention-style fusion yet.

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

## Next steps (Phase 2/3, not yet implemented)

- Tiled shared-memory matmul; memory coalescing / bank-conflict fixes.
- `__shfl_sync` warp-level reductions for softmax and layernorm.
- Fused bias+GELU and QK^T+softmax kernels.
- Nsight Compute / Nsight Systems profiling on Sol; roofline analysis (compute- vs memory-bound, %peak FLOPs/bandwidth).
- Naive → tiled → fused speedup chart vs native PyTorch and cuBLAS.
