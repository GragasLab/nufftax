#!/usr/bin/env python
"""
GPU benchmark for scatter-add strategies.

Run on zay cluster with:
    srun --partition=gpu_p5 --gres=gpu:1 --time=00:30:00 \
         python benchmarks/bench_gpu_scatter.py

This script compares add.at vs segment_sum on GPU to determine
which is faster for NUFFT spreading operations.
"""

import json
import time
from datetime import datetime

import jax
import jax.numpy as jnp


# Print device info
print("=" * 70)
print("GPU SCATTER-ADD BENCHMARK")
print("=" * 70)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print()


def benchmark(fn, args, n_warmup=5, n_runs=100, name=""):
    """Benchmark a function with proper GPU synchronization."""
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        result = fn(*args)
        jax.block_until_ready(result)
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"{name:45s}: {elapsed:.4f} ms")
    return elapsed


# =============================================================================
# Strategy 1: add.at (current in many implementations)
# =============================================================================


def make_scatter_add_at(grid_size):
    @jax.jit
    def scatter_add_at(indices, values):
        grid = jnp.zeros(grid_size, dtype=values.dtype)
        return grid.at[indices].add(values)

    return scatter_add_at


# =============================================================================
# Strategy 2: segment_sum (new optimization)
# =============================================================================


def make_scatter_segment_sum(grid_size):
    @jax.jit
    def scatter_segment_sum(indices, values):
        return jax.ops.segment_sum(values, indices, num_segments=grid_size)

    return scatter_segment_sum


# =============================================================================
# Run benchmarks
# =============================================================================


def run_single_benchmark(M, grid_size, dtype=jnp.float32, n_runs=100):
    """Run benchmark for a single configuration."""
    key = jax.random.PRNGKey(42)
    indices = jax.random.randint(key, (M,), 0, grid_size)
    values = jax.random.normal(jax.random.PRNGKey(43), (M,), dtype=dtype)

    scatter_add_at = make_scatter_add_at(grid_size)
    scatter_segment_sum = make_scatter_segment_sum(grid_size)

    time_add_at = benchmark(
        scatter_add_at,
        (indices, values),
        n_runs=n_runs,
        name=f"add.at (M={M:,}, grid={grid_size:,})",
    )
    time_segment_sum = benchmark(
        scatter_segment_sum,
        (indices, values),
        n_runs=n_runs,
        name=f"segment_sum (M={M:,}, grid={grid_size:,})",
    )

    speedup = time_add_at / time_segment_sum
    winner = "segment_sum" if speedup > 1 else "add.at"
    print(f"  -> Speedup: {speedup:.2f}x ({winner} is faster)")
    print()

    return {
        "M": M,
        "grid_size": grid_size,
        "dtype": str(dtype),
        "add_at_ms": time_add_at,
        "segment_sum_ms": time_segment_sum,
        "speedup": speedup,
        "winner": winner,
    }


def run_nufft_realistic_benchmark(M, nspread, grid_size, dtype=jnp.float32, n_runs=100):
    """Run benchmark with realistic NUFFT spreading pattern."""
    print(f"\nRealistic NUFFT pattern: M={M:,} x nspread={nspread} = {M * nspread:,} contributions")
    print(f"Grid size: {grid_size:,}")

    key = jax.random.PRNGKey(42)
    # Each point contributes to nspread consecutive grid points
    base_indices = jax.random.randint(key, (M,), 0, grid_size - nspread)
    offsets = jnp.arange(nspread)
    indices = (base_indices[:, None] + offsets[None, :]).ravel()
    indices = indices % grid_size

    values = jax.random.normal(jax.random.PRNGKey(43), (M * nspread,), dtype=dtype)

    scatter_add_at = make_scatter_add_at(grid_size)
    scatter_segment_sum = make_scatter_segment_sum(grid_size)

    time_add_at = benchmark(
        scatter_add_at,
        (indices, values),
        n_runs=n_runs,
        name=f"add.at (NUFFT M={M:,}, nspread={nspread})",
    )
    time_segment_sum = benchmark(
        scatter_segment_sum,
        (indices, values),
        n_runs=n_runs,
        name=f"segment_sum (NUFFT M={M:,}, nspread={nspread})",
    )

    speedup = time_add_at / time_segment_sum
    winner = "segment_sum" if speedup > 1 else "add.at"
    print(f"  -> Speedup: {speedup:.2f}x ({winner} is faster)")

    return {
        "M": M,
        "nspread": nspread,
        "total_contributions": M * nspread,
        "grid_size": grid_size,
        "dtype": str(dtype),
        "add_at_ms": time_add_at,
        "segment_sum_ms": time_segment_sum,
        "speedup": speedup,
        "winner": winner,
    }


def main():
    results = {
        "timestamp": datetime.now().isoformat(),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "backend": jax.default_backend(),
        "benchmarks": [],
        "nufft_benchmarks": [],
    }

    # Basic scatter benchmarks
    print("\n" + "=" * 70)
    print("BASIC SCATTER-ADD BENCHMARKS")
    print("=" * 70)

    configs = [
        # (M, grid_size)
        (100_000, 512),  # 1D NUFFT typical
        (100_000, 2048),  # 1D NUFFT larger
        (100_000, 65536),  # 2D NUFFT (256x256)
        (100_000, 262144),  # 2D NUFFT (512x512)
        (1_000_000, 512),  # Many points, small grid
        (1_000_000, 2048),  # Many points, medium grid
        (1_000_000, 65536),  # Many points, large grid
    ]

    for M, grid_size in configs:
        result = run_single_benchmark(M, grid_size, n_runs=100)
        results["benchmarks"].append(result)

    # Realistic NUFFT spreading patterns
    print("\n" + "=" * 70)
    print("REALISTIC NUFFT SPREADING PATTERNS")
    print("=" * 70)

    nufft_configs = [
        # (M, nspread, grid_size)
        (100_000, 8, 512),  # 1D typical
        (100_000, 8, 2048),  # 1D larger grid
        (100_000, 8, 65536),  # 2D (256x256)
        (1_000_000, 8, 512),  # Many points 1D
        (1_000_000, 8, 2048),  # Many points larger
        (100_000, 12, 512),  # Higher precision (larger kernel)
        (100_000, 16, 512),  # Very high precision
    ]

    for M, nspread, grid_size in nufft_configs:
        result = run_nufft_realistic_benchmark(M, nspread, grid_size, n_runs=100)
        results["nufft_benchmarks"].append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    add_at_wins = sum(1 for r in results["benchmarks"] if r["winner"] == "add.at")
    segment_sum_wins = sum(1 for r in results["benchmarks"] if r["winner"] == "segment_sum")
    print(f"\nBasic benchmarks: add.at wins {add_at_wins}, segment_sum wins {segment_sum_wins}")

    add_at_wins_nufft = sum(1 for r in results["nufft_benchmarks"] if r["winner"] == "add.at")
    segment_sum_wins_nufft = sum(1 for r in results["nufft_benchmarks"] if r["winner"] == "segment_sum")
    print(f"NUFFT benchmarks: add.at wins {add_at_wins_nufft}, segment_sum wins {segment_sum_wins_nufft}")

    avg_speedup_basic = sum(r["speedup"] for r in results["benchmarks"]) / len(results["benchmarks"])
    avg_speedup_nufft = sum(r["speedup"] for r in results["nufft_benchmarks"]) / len(results["nufft_benchmarks"])
    print("\nAverage speedup (segment_sum vs add.at):")
    print(f"  Basic: {avg_speedup_basic:.2f}x")
    print(f"  NUFFT: {avg_speedup_nufft:.2f}x")

    if avg_speedup_basic > 1 and avg_speedup_nufft > 1:
        print("\n*** RECOMMENDATION: Use segment_sum ***")
    elif avg_speedup_basic < 1 and avg_speedup_nufft < 1:
        print("\n*** RECOMMENDATION: Revert to add.at ***")
    else:
        print("\n*** RECOMMENDATION: Results mixed, need more analysis ***")

    # Save results to file
    output_file = f"gpu_scatter_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
