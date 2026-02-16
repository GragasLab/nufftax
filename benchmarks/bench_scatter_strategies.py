"""
Benchmark different scatter-add strategies for NUFFT spreading.

Tests:
1. jnp.add.at (current implementation)
2. jax.lax.scatter_add with indices_are_sorted=True
3. jax.ops.segment_sum
4. Dense matrix multiplication (baseline for small grids)
"""

import time

import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Device: {jax.devices()[0]}")
print()


def benchmark(fn, args, n_warmup=3, n_runs=50, name=""):
    """Benchmark a function."""
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

    print(f"{name:40s}: {elapsed:.3f} ms")
    return elapsed


# =============================================================================
# Strategy 1: Current implementation (jnp.add.at)
# =============================================================================


def make_scatter_add_at(grid_size):
    @jax.jit
    def scatter_add_at(indices, values):
        """Current implementation using jnp.add.at"""
        grid = jnp.zeros(grid_size, dtype=values.dtype)
        return grid.at[indices].add(values)

    return scatter_add_at


# =============================================================================
# Strategy 2: jax.lax.scatter_add with sorted indices
# =============================================================================


def make_scatter_add_lax_sorted(grid_size):
    @jax.jit
    def scatter_add_lax_sorted(indices, values):
        """Using jax.lax.scatter_add with indices_are_sorted hint."""
        grid = jnp.zeros(grid_size, dtype=values.dtype)
        # Sort indices and reorder values accordingly
        sort_idx = jnp.argsort(indices)
        sorted_indices = indices[sort_idx]
        sorted_values = values[sort_idx]
        return grid.at[sorted_indices].add(sorted_values, indices_are_sorted=True)

    return scatter_add_lax_sorted


def make_scatter_add_lax_presorted(grid_size):
    @jax.jit
    def scatter_add_lax_presorted(sorted_indices, sorted_values):
        """Assuming indices are already sorted (skip sort overhead)."""
        grid = jnp.zeros(grid_size, dtype=sorted_values.dtype)
        return grid.at[sorted_indices].add(sorted_values, indices_are_sorted=True)

    return scatter_add_lax_presorted


# =============================================================================
# Strategy 3: segment_sum
# =============================================================================


def make_scatter_segment_sum(grid_size):
    @jax.jit
    def scatter_segment_sum(indices, values):
        """Using jax.ops.segment_sum."""
        return jax.ops.segment_sum(values, indices, num_segments=grid_size)

    return scatter_segment_sum


def make_scatter_segment_sum_sorted(grid_size):
    @jax.jit
    def scatter_segment_sum_sorted(indices, values):
        """Using segment_sum with sorted indices hint."""
        sort_idx = jnp.argsort(indices)
        sorted_indices = indices[sort_idx]
        sorted_values = values[sort_idx]
        return jax.ops.segment_sum(sorted_values, sorted_indices, num_segments=grid_size, indices_are_sorted=True)

    return scatter_segment_sum_sorted


# =============================================================================
# Strategy 4: Dense matrix multiplication (for small grids only)
# =============================================================================


def make_scatter_dense_matmul(grid_size):
    @jax.jit
    def scatter_dense_matmul(indices, values):
        """
        Convert to one-hot and use matmul.
        Only efficient for small grid_size.
        """
        # Create one-hot matrix: (M,) -> (M, grid_size)
        one_hot = jax.nn.one_hot(indices, grid_size, dtype=values.dtype)
        # Matmul: (grid_size, M) @ (M,) -> (grid_size,)
        return one_hot.T @ values

    return scatter_dense_matmul


# =============================================================================
# Run benchmarks
# =============================================================================


def run_benchmarks(M, grid_size, dtype=jnp.float32):
    """Run all benchmarks for given problem size."""
    print(f"\n{'=' * 60}")
    print(f"M = {M:,} values, grid_size = {grid_size:,}")
    print(f"{'=' * 60}")

    key = jax.random.PRNGKey(42)
    indices = jax.random.randint(key, (M,), 0, grid_size)
    values = jax.random.normal(jax.random.PRNGKey(43), (M,), dtype=dtype)

    # Pre-sort for sorted benchmarks
    sort_idx = jnp.argsort(indices)
    sorted_indices = indices[sort_idx]
    sorted_values = values[sort_idx]

    # Create functions with static grid_size
    scatter_add_at = make_scatter_add_at(grid_size)
    scatter_add_lax_sorted = make_scatter_add_lax_sorted(grid_size)
    scatter_add_lax_presorted = make_scatter_add_lax_presorted(grid_size)
    scatter_segment_sum = make_scatter_segment_sum(grid_size)
    scatter_segment_sum_sorted = make_scatter_segment_sum_sorted(grid_size)

    results = {}

    # Baseline
    results["add.at"] = benchmark(scatter_add_at, (indices, values), name="jnp.add.at (current)")

    # With sorting
    results["add.at+sort"] = benchmark(
        scatter_add_lax_sorted, (indices, values), name="add.at + sort + indices_are_sorted"
    )

    # Pre-sorted (optimal case)
    results["add.at presorted"] = benchmark(
        scatter_add_lax_presorted,
        (sorted_indices, sorted_values),
        name="add.at presorted + indices_are_sorted",
    )

    # segment_sum
    results["segment_sum"] = benchmark(scatter_segment_sum, (indices, values), name="segment_sum")

    # segment_sum sorted
    results["segment_sum+sort"] = benchmark(
        scatter_segment_sum_sorted, (indices, values), name="segment_sum + sort + sorted hint"
    )

    # Dense matmul (only for small grids)
    if grid_size <= 4096:
        scatter_dense_matmul = make_scatter_dense_matmul(grid_size)
        results["dense_matmul"] = benchmark(scatter_dense_matmul, (indices, values), name="dense matmul (one-hot)")

    # Print summary
    baseline = results["add.at"]
    print("\nRelative to baseline (add.at):")
    for name, time_ms in results.items():
        speedup = baseline / time_ms
        print(f"  {name:35s}: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    # Test different problem sizes

    # Small grid (1D NUFFT typical)
    run_benchmarks(M=100_000, grid_size=512)

    # Medium grid
    run_benchmarks(M=100_000, grid_size=2048)

    # Large grid (2D flattened)
    run_benchmarks(M=100_000, grid_size=65536)  # 256x256

    # Many points
    run_benchmarks(M=1_000_000, grid_size=2048)

    # Realistic NUFFT: M points with nspread=8 contributions each
    # This simulates the actual spreading pattern
    print("\n" + "=" * 60)
    print("REALISTIC NUFFT SPREADING PATTERN")
    print("=" * 60)

    M = 100_000
    nspread = 8
    grid_size = 512

    key = jax.random.PRNGKey(42)
    # Each point contributes to nspread grid points
    base_indices = jax.random.randint(key, (M,), 0, grid_size - nspread)
    offsets = jnp.arange(nspread)
    # Full indices: (M, nspread) -> (M * nspread,)
    indices = (base_indices[:, None] + offsets[None, :]).ravel()
    indices = indices % grid_size

    values = jax.random.normal(jax.random.PRNGKey(43), (M * nspread,), dtype=jnp.float32)

    print(f"\nM = {M:,} points x nspread = {nspread} = {M * nspread:,} contributions")
    print(f"Grid size = {grid_size}")

    scatter_add_at = make_scatter_add_at(grid_size)
    scatter_segment_sum = make_scatter_segment_sum(grid_size)

    benchmark(scatter_add_at, (indices, values), name="add.at (realistic)")
    benchmark(scatter_segment_sum, (indices, values), name="segment_sum (realistic)")
