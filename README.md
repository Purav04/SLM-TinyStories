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
- `profiling/profile_kernels.py` — isolated, NVTX-annotated kernel calls for Nsight Compute / Nsight Systems (Phase 3).
- `profiling/roofline.py` — analytical roofline plot (`ncu`'s hardware counters are blocked on Sol for non-admin users; this computes idealized arithmetic intensity from each kernel's own access pattern instead). Pure Python + matplotlib, no GPU needed.

## Status

**Phase 1 (naive kernels) — verified on Sol (A100).** `pytest tests/test_ops.py -v`
passed 7/7: matmul against all transpose combinations, `CudaLinear`/
`cuda_softmax`/`CudaLayerNorm` forward+backward against their
`torch.nn.functional` equivalents, and a full `GPT` forward+backward
matching between the native and CUDA-kernel code paths.

**Phase 2 (tiling, coalescing, warp shuffles) — verified on Sol (A100),
12/12 tests passing.**

- `matmul_tiled` (`csrc/matmul_kernel.cu`): shared-memory tiled GEMM, `TILE_DIM=32`. Every global-memory tile load has `threadIdx.x` mapped to whichever dimension is physically contiguous — regardless of the logical transpose flag — so all four transpose combinations load coalesced; the transpose is instead handled by how the compute loop indexes back into shared memory. Shared tiles are padded (`[TILE][TILE+1]`) to avoid bank conflicts on the transposed reads. `CudaLinear` now uses this by default; the Phase 1 naive kernel (`matmul`) is kept for the benchmark comparison. Measured on Sol at 1024³: **1.28x faster than naive, ~23% of cuBLAS throughput** (cuBLAS uses TF32 tensor cores on A100 — a different, much faster execution path than our plain-CUDA-core kernel; closing more of that gap needs register blocking / vectorized loads, a further optimization pass).
- Softmax and LayerNorm's `dx`/mean/var row reductions (`csrc/common.h`: `blockReduceSum`/`blockReduceMax`) now use `__shfl_down_sync` warp-shuffle reduction instead of Phase 1's shared-memory tree. Measured: **softmax is now ~8% faster than `F.softmax`.**
- LayerNorm's `dweight`/`dbias` backward: tried to "fix" the Phase 1 `atomicAdd` (8192 rows atomically adding into the same 768-wide buffer looks like an obvious bottleneck) two different ways, and both measured *worse* than the atomics on Sol — a one-thread-per-column kernel with no contention but only 768 threads total (idling most of a 108-SM chip): 2.36ms, 7x worse; delegating to ATen's `.sum(0)`: 0.40ms, still ~20% worse. Kept the original atomics (measured 0.33ms) — the buffer they contend on is tiny (3KB) and stays resident in L2, so contention costs far less here than expected, and the fused single-kernel-pass-with-dx beat both alternatives. Documented as a deliberate "measured, didn't help, reverted" decision in `csrc/layernorm_kernel.cu`.
- `benchmarks/bench.py` gained `bench_matmul_variants()`: naive vs tiled vs cuBLAS throughput (GFLOP/s) on a raw GEMM, no autograd overhead — the number for the eventual speedup chart.
- New tests: `TestMatmulTiled` covers all four transpose combinations (Phase 1 only covered three) at non-tile-multiple shapes, to exercise the boundary-padding logic, plus a direct naive-vs-tiled equality check.

Note on `CudaLinear`'s end-to-end benchmark: it's currently *slower* than
`nn.Linear` (3 separate ~23%-of-cuBLAS GEMM launches vs. torch calling
cuBLAS directly with near-zero overhead). That's expected at this stage —
the project's own goal is "Nx over naive, Y% of cuBLAS" per kernel, not
beating a hand-tuned vendor library's fused path outright.

Not yet started:

- Fused kernels (bias+GELU, QK^T+softmax) — the attention QK^T matmul and the softmax over attention scores are still two separate ops.
- Register blocking / vectorized loads to push the tiled matmul closer to cuBLAS throughput.

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

## Phase 3: profiling

**Nsight Compute is blocked on this cluster for non-admin users.** Every
metric — including the single most basic one, `gpu__time_duration.sum` —
fails immediately with `ResourceUnavailable` / "Profiling failed because a
driver resource was unavailable," on both a MIG-partitioned node and a
full non-MIG A100. The log (`Failed to create counter availability image
(error = 20)`) points to the `NVreg_RestrictProfilingToAdminUsers` driver
flag, a common HPC-cluster security policy — not something fixable by
node choice or metric selection. Filing a ticket with Sol's HPC support is
the real fix; not pursued further here since it's outside this project's
control.

**Nsight Systems works** (`nsys profile` + `nsys stats --report
cuda_gpu_kern_sum`) — it uses CUPTI's activity-tracing API rather than raw
hardware counters, a much less privileged mechanism. Reproduce with:

```bash
python profiling/profile_kernels.py
nsys profile -o profiling/nsys_report --force-overwrite=true python profiling/profile_kernels.py
nsys stats --force-export=true --report cuda_gpu_kern_sum profiling/nsys_report.nsys-rep
```

Measured on Sol, full non-MIG A100-SXM4-80GB:

| kernel | 8192×768×768 | 4096³ | degradation |
|---|---|---|---|
| naive matmul | 2792 GFLOP/s | 2265 GFLOP/s | **-19%** |
| tiled matmul | 3757 GFLOP/s | 3714 GFLOP/s | **-1%** |
| cuBLAS (`ampere_sgemm_*`) | 15,766 GFLOP/s | 17,233 GFLOP/s | +9% |

Two real findings out of this:

1. **The L2-cache-masking effect is confirmed on clean, non-MIG hardware.** 4096³'s ~192MB working set exceeds the A100's 40MB L2; naive degrades 19% once it can no longer lean on cache to cover for its lack of memory reuse, while tiled — which actually reuses shared-memory tiles — barely moves. This is real evidence the tiling is doing its job, independent of the raw throughput numbers.
2. **cuBLAS isn't using tensor cores here** — the kernel names (`ampere_sgemm_128x32_nn`, `ampere_sgemm_128x64_nn`) are plain FP32 CUDA-core kernels, not tensor-core ones, and it picks a *different* kernel per problem shape (something a single fixed hand-written kernel can't do). So cuBLAS's ~4.5x edge over our tiled kernel at 4096³ (17,233 vs 3,714 GFLOP/s, tiled = 21.5% of cuBLAS) is pure classical-GEMM engineering — register blocking, double buffering, shape-adaptive tile selection — not a fundamentally different hardware path. That's exactly what register blocking + vectorized loads (the next planned optimization) targets.
3. **Softmax/LayerNorm forward kernels, isolated, are 2-2.4x slower than PyTorch's** (softmax: 268µs vs 111µs; layernorm: 113µs vs 52µs) — despite the earlier end-to-end (forward+backward) benchmark showing our softmax *ahead*. Isolating the forward kernel via `nsys` caught something the wall-clock number hid: our forward kernel itself isn't actually faster, backward/dispatch overhead on PyTorch's side was carrying that earlier comparison. Reported here rather than the more flattering number, since that's the point of profiling.

**Roofline (analytical, not `ncu`-measured):** since hardware counters
aren't available, `profiling/roofline.py` computes each matmul kernel's
*idealized* arithmetic intensity from its own access pattern (naive:
0.25 FLOP/byte, no reuse; tiled: 8.0 FLOP/byte, ~32x reuse from
`TILE_DIM`) and plots it against the A100's published peak compute (19.5
TFLOP/s FP32) and bandwidth (2039 GB/s) — the standard fallback when
`ncu`'s measured DRAM/L2 traffic isn't available. Run with
`python profiling/roofline.py` (pure arithmetic + matplotlib, no GPU
needed). Findings:

- Tiled's idealized intensity (8.0) sits just under the A100's ridge point (9.56 FLOP/byte) — close to the compute/memory boundary. But achieved throughput (3714 GFLOP/s at 4096³) is only **23% of tiled's own idealized memory-bound ceiling** (16,312 GFLOP/s). That gap means tiled *isn't* bandwidth-limited in practice — something else (low occupancy, one output element per thread with no register blocking, scalar rather than vectorized `float4` loads) is the real bottleneck. Directly motivates the next optimization pass.
- Naive achieves **~4.4x more** than its own idealized no-reuse ceiling (2265 vs 510 GFLOP/s) — not a contradiction, it's L2 cache doing real work the idealized model doesn't account for. The idealized intensity model is a cache-blind lower bound, not a hard ceiling; this is consistent with, and further confirms, finding #1 above.

## Next steps

- Register blocking / vectorized (`float4`) loads for the tiled matmul — the clearest next lever, per the roofline finding above (tiled is nowhere near its own memory-bound ceiling, so more parallelism/ILP per thread is the fix, not further bandwidth work).
- Fused bias+GELU and QK^T+softmax kernels (Phase 2 fusion work, not yet started).
- If Sol support re-enables `ncu`, re-run `--set roofline` for measured (not idealized) arithmetic intensity and compare against this analytical version.
- Naive → tiled → fused speedup chart vs native PyTorch and cuBLAS, folding in the fused-kernel numbers once built.
