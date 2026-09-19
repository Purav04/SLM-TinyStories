"""Analytical roofline for the matmul kernels — built without Nsight
Compute, since ncu's hardware performance counters are blocked at the
driver level on this cluster for non-admin users (confirmed: even the
single most basic metric, gpu__time_duration.sum, fails immediately with
"ResourceUnavailable" on a full, non-MIG A100 — this is a permissions
policy, not something fixable by kernel/metric choice or node selection).

Standard substitute when hardware counters aren't available: compute each
kernel's *idealized* arithmetic intensity (FLOPs per byte) directly from
its own memory-access pattern, and plot that against the GPU's published
peak compute and peak memory bandwidth. This is the original Roofline
model (Williams et al.) applied analytically instead of from measured
DRAM/L2 traffic — less precise than ncu's actual counters, but honest and
still answers the real question: is each kernel bandwidth-limited or
compute-limited, and how far is it from its own ceiling.

All GFLOP/s numbers below are measured (nsys, cuda_gpu_kern_sum) on Sol,
a full non-MIG A100-SXM4-80GB, from profiling/profile_kernels.py's two
matmul shapes:
  - "model": 8192x768x768 (matches CudaLinear's real usage in the GPT
    model) — small enough to mostly fit in the A100's 40MB L2 cache.
  - "large": 4096x4096x4096, ~192MB working set — exceeds L2, so this is
    the size that reflects each kernel's *true* intensity-limited
    behavior without cache reuse partially masking it.

Run locally (no GPU needed — this is pure arithmetic + matplotlib):
    python profiling/roofline.py
"""
import matplotlib.pyplot as plt
import numpy as np

# --- A100 SXM4 80GB published peak specs ---
PEAK_COMPUTE_GFLOPS = 19_500.0  # FP32, non-Tensor-Core
PEAK_BW_GBS = 2_039.0           # HBM2e
RIDGE_POINT = PEAK_COMPUTE_GFLOPS / PEAK_BW_GBS  # ~9.56 FLOP/byte

# --- Idealized arithmetic intensity, from each kernel's own access pattern ---
# Naive: every thread re-reads its full row of A and column of B from
# global memory with zero reuse across threads. bytes = 2 reads * K * 4B
# per output element; FLOPs = 2*K per output element (one multiply-add).
# intensity = 2K / (2*K*4) = 0.25 FLOP/byte, independent of K.
NAIVE_INTENSITY = 0.25

# Tiled: each TILE_DIM x TILE_DIM tile of A and B is loaded once into
# shared memory and reused TILE_DIM times by the threads that need it,
# cutting global traffic by ~TILE_DIM vs naive.
TILE_DIM = 32
TILED_INTENSITY = NAIVE_INTENSITY * TILE_DIM  # 8.0 FLOP/byte

# --- Measured achieved throughput (GFLOP/s), nsys on Sol, full A100 ---
MEASURED = {
    "naive (8192x768x768)": (NAIVE_INTENSITY, 2792.0),
    "naive (4096³)": (NAIVE_INTENSITY, 2264.7),
    "tiled (8192x768x768)": (TILED_INTENSITY, 3757.1),
    "tiled (4096³)": (TILED_INTENSITY, 3714.0),
}
# cuBLAS's exact algorithm/intensity isn't characterized (proprietary,
# and its kernel choice literally changes with problem shape — nsys
# showed two different kernel names, ampere_sgemm_128x32_nn and
# ampere_sgemm_128x64_nn, for the two sizes). Shown as a reference line
# at its achieved throughput rather than a specific (intensity, GFLOP/s)
# point, since we can't honestly place it on the x-axis.
CUBLAS_GFLOPS = (15_765.6 + 17_232.9) / 2  # avg of the two measured sizes


def roofline_ceiling(intensity):
    return min(PEAK_COMPUTE_GFLOPS, intensity * PEAK_BW_GBS)


def main():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    x = np.logspace(-2, 2.5, 400)
    y = np.minimum(PEAK_COMPUTE_GFLOPS, x * PEAK_BW_GBS)
    ax.plot(x, y, color="#2b6cb0", lw=2, label="A100 FP32 roofline (19.5 TFLOP/s, 2039 GB/s)")
    ax.axvline(RIDGE_POINT, color="#a0aec0", ls=":", lw=1)
    ax.text(RIDGE_POINT * 0.85, PEAK_COMPUTE_GFLOPS * 0.02, f"ridge point\n{RIDGE_POINT:.2f} FLOP/byte",
             fontsize=8, color="#718096", ha="right")

    ax.axhline(CUBLAS_GFLOPS, color="#38a169", ls="--", lw=1.5,
               label=f"cuBLAS achieved (avg {CUBLAS_GFLOPS:,.0f} GFLOP/s — intensity not characterized)")

    markers = {"naive": "o", "tiled": "^"}
    colors = {"naive": "#e53e3e", "tiled": "#dd6b20"}
    # Per-point label offsets (dx, dy in points) — the two points of each
    # kind share an x (same idealized intensity), so their labels would
    # otherwise land on top of each other.
    label_offsets = {
        "naive (8192x768x768)": (10, 4),
        "naive (4096³)": (10, -10),
        "tiled (8192x768x768)": (10, 6),
        "tiled (4096³)": (10, -14),
    }
    seen_label = set()
    for name, (intensity, gflops) in MEASURED.items():
        kind = "naive" if "naive" in name else "tiled"
        label = f"{kind} kernel (measured)" if kind not in seen_label else None
        seen_label.add(kind)
        ax.scatter(intensity, gflops, marker=markers[kind], color=colors[kind], s=90, zorder=5, label=label)
        ax.annotate(name, (intensity, gflops), textcoords="offset points",
                    xytext=label_offsets[name], fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achieved throughput (GFLOP/s)")
    ax.set_title("SLM-TinyStories matmul kernels vs A100 FP32 roofline\n(analytical intensity — Nsight Compute counters blocked on this cluster)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", ls="-", alpha=0.15)

    fig.tight_layout()
    out_path = "profiling/roofline.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    print()
    print(f"Ridge point: {RIDGE_POINT:.2f} FLOP/byte")
    print(f"naive intensity {NAIVE_INTENSITY:.2f} -> idealized ceiling {roofline_ceiling(NAIVE_INTENSITY):.1f} GFLOP/s "
          f"(deeply memory-bound in the idealized no-reuse model)")
    print(f"tiled intensity {TILED_INTENSITY:.2f} -> idealized ceiling {roofline_ceiling(TILED_INTENSITY):.1f} GFLOP/s")
    print(f"tiled achieved (4096³) is {3714.0 / roofline_ceiling(TILED_INTENSITY) * 100:.1f}% of its own "
          f"idealized ceiling -> not bandwidth-limited in practice; the gap is occupancy/ILP/vectorization, "
          f"exactly what register blocking + float4 loads target next.")
    print(f"naive achieved (4096³) is {2264.7 / roofline_ceiling(NAIVE_INTENSITY) * 100:.0f}% OF its idealized "
          f"no-reuse ceiling -> L2 cache is doing real work compensating for the naive access pattern; the "
          f"idealized model is a cache-blind lower bound, not a hard ceiling.")


if __name__ == "__main__":
    main()
