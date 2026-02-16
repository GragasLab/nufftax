"""
Benchmark JAX-specific optimizations for NUFFT.

Tests:
1. JIT compilation boundaries
2. Gradient checkpointing (jax.checkpoint/remat)
3. donate_argnums for buffer reuse
4. vmap vs manual batching

References:
- https://docs.jax.dev/en/latest/gpu_performance_tips.html
- https://docs.jax.dev/en/latest/gradient-checkpointing.html
"""

import time
from functools import partial

import jax
import jax.numpy as jnp

# Enable x64 for accuracy
jax.config.update("jax_enable_x64", False)  # Use float32 for speed


def benchmark_function(func, args, n_warmup=3, n_runs=10, name=""):
    """Benchmark a function with warmup."""
    # Warmup
    for _ in range(n_warmup):
        result = func(*args)
        jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - start) * 1000)

    mean_ms = float(jnp.mean(jnp.array(times)))
    std_ms = float(jnp.std(jnp.array(times)))
    return {"name": name, "mean_ms": mean_ms, "std_ms": std_ms}


# =============================================================================
# Test 1: JIT compilation of full NUFFT vs partial JIT
# =============================================================================


def test_jit_boundaries():
    """Test full JIT vs partial JIT compilation."""
    from nufftax.transforms.nufft1 import nufft1d1

    print("\n" + "=" * 70)
    print("Test 1: JIT Compilation Boundaries")
    print("=" * 70)

    for M in [10_000, 100_000]:
        for N in [128, 512]:
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex64)

            # Version 1: No explicit JIT (relies on internal JIT)
            def nufft_no_jit(x, c, n_modes=N):
                return nufft1d1(x, c, n_modes, eps=1e-6)

            # Version 2: Full JIT wrapper
            @jax.jit
            def nufft_full_jit(x, c, n_modes=N):
                return nufft1d1(x, c, n_modes, eps=1e-6)

            stats_no_jit = benchmark_function(nufft_no_jit, (x, c), name="No outer JIT")
            stats_jit = benchmark_function(nufft_full_jit, (x, c), name="Full JIT")

            speedup = stats_no_jit["mean_ms"] / stats_jit["mean_ms"]
            print(
                f"M={M:>6}, N={N:>4}: no_jit={stats_no_jit['mean_ms']:.3f}ms, "
                f"full_jit={stats_jit['mean_ms']:.3f}ms, speedup={speedup:.2f}x"
            )


# =============================================================================
# Test 2: Gradient checkpointing
# =============================================================================


def test_gradient_checkpointing():
    """Test memory vs compute tradeoff with gradient checkpointing."""
    from nufftax.transforms.nufft1 import nufft1d1

    print("\n" + "=" * 70)
    print("Test 2: Gradient Checkpointing")
    print("=" * 70)

    for M in [10_000, 50_000]:
        N = 256

        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = (
            jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        ).astype(jnp.complex64)

        # Loss function
        def loss_fn(x, c, n_modes=N):
            result = nufft1d1(x, c, n_modes, eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2)

        # Version 1: Standard gradient
        @jax.jit
        def grad_standard(x, c):
            return jax.grad(loss_fn, argnums=(0, 1))(x, c)

        # Version 2: Checkpointed gradient
        @jax.jit
        def grad_checkpointed(x, c):
            @jax.checkpoint
            def checkpointed_loss(x, c):
                return loss_fn(x, c)

            return jax.grad(checkpointed_loss, argnums=(0, 1))(x, c)

        stats_standard = benchmark_function(grad_standard, (x, c), name="Standard grad")
        stats_checkpoint = benchmark_function(grad_checkpointed, (x, c), name="Checkpointed")

        speedup = stats_standard["mean_ms"] / stats_checkpoint["mean_ms"]
        print(
            f"M={M:>6}: standard={stats_standard['mean_ms']:.3f}ms, "
            f"checkpoint={stats_checkpoint['mean_ms']:.3f}ms, ratio={speedup:.2f}x"
        )


# =============================================================================
# Test 3: vmap vs manual batching
# =============================================================================


def test_vmap_batching():
    """Test vmap automatic batching vs built-in batching."""
    from nufftax.transforms.nufft1 import nufft1d1

    print("\n" + "=" * 70)
    print("Test 3: vmap vs Manual Batching")
    print("=" * 70)

    M = 10_000
    N = 256

    for n_batch in [4, 16, 64]:
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c_batch = (
            jax.random.normal(jax.random.PRNGKey(43), (n_batch, M))
            + 1j * jax.random.normal(jax.random.PRNGKey(44), (n_batch, M))
        ).astype(jnp.complex64)

        # Version 1: Built-in batching (pass 2D c array)
        @jax.jit
        def nufft_builtin_batch(x, c):
            return nufft1d1(x, c, N, eps=1e-6)

        # Version 2: vmap over c axis
        @jax.jit
        def nufft_vmap(x, c):
            return jax.vmap(lambda ci: nufft1d1(x, ci, N, eps=1e-6))(c)

        stats_builtin = benchmark_function(nufft_builtin_batch, (x, c_batch), name="Built-in batch")
        stats_vmap = benchmark_function(nufft_vmap, (x, c_batch), name="vmap")

        speedup = stats_vmap["mean_ms"] / stats_builtin["mean_ms"]
        print(
            f"n_batch={n_batch:>3}: builtin={stats_builtin['mean_ms']:.3f}ms, "
            f"vmap={stats_vmap['mean_ms']:.3f}ms, builtin/vmap={speedup:.2f}x"
        )


# =============================================================================
# Test 4: Static vs traced arguments
# =============================================================================


def test_static_args():
    """Test effect of static arguments on recompilation."""
    from nufftax.transforms.nufft1 import nufft1d1

    print("\n" + "=" * 70)
    print("Test 4: Static Arguments Effect")
    print("=" * 70)

    M = 50_000

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = (jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))).astype(
        jnp.complex64
    )

    # Test with different N values (causes recompilation)
    print("Testing recompilation overhead with different N values:")

    # N must be static since it determines output shape
    @partial(jax.jit, static_argnums=(2,))
    def nufft_jit(x, c, N):
        return nufft1d1(x, c, N, eps=1e-6)

    total_time_first = 0
    total_time_cached = 0

    for N in [64, 128, 256, 512]:
        # First call (includes compilation)
        start = time.perf_counter()
        result = nufft_jit(x, c, N)
        jax.block_until_ready(result)
        first_call = (time.perf_counter() - start) * 1000

        # Second call (cached)
        start = time.perf_counter()
        result = nufft_jit(x, c, N)
        jax.block_until_ready(result)
        cached_call = (time.perf_counter() - start) * 1000

        total_time_first += first_call
        total_time_cached += cached_call
        print(f"  N={N:>4}: first={first_call:.1f}ms, cached={cached_call:.3f}ms")

    print(f"\nTotal: first={total_time_first:.1f}ms, cached={total_time_cached:.1f}ms")


# =============================================================================
# Test 5: donate_argnums for buffer reuse
# =============================================================================


def test_donate_argnums():
    """Test if donate_argnums can help with buffer reuse."""
    from nufftax.transforms.nufft1 import nufft1d1

    print("\n" + "=" * 70)
    print("Test 5: donate_argnums Buffer Reuse")
    print("=" * 70)

    M = 100_000
    N = 512

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = (jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))).astype(
        jnp.complex64
    )

    # Version 1: No donation
    @jax.jit
    def nufft_no_donate(x, c):
        return nufft1d1(x, c, N, eps=1e-6)

    # Version 2: With donation (donate c since it's not needed after)
    # Note: donate_argnums only works if the donated buffer isn't used after
    @partial(jax.jit, donate_argnums=(1,))
    def nufft_donate(x, c):
        return nufft1d1(x, c, N, eps=1e-6)

    # Benchmark (need fresh c each time for donation)
    n_runs = 10

    # No donate
    times_no_donate = []
    for _ in range(n_runs):
        c_copy = c.copy()
        start = time.perf_counter()
        result = nufft_no_donate(x, c_copy)
        jax.block_until_ready(result)
        times_no_donate.append((time.perf_counter() - start) * 1000)

    # With donate
    times_donate = []
    for _ in range(n_runs):
        c_copy = c.copy()
        start = time.perf_counter()
        result = nufft_donate(x, c_copy)
        jax.block_until_ready(result)
        times_donate.append((time.perf_counter() - start) * 1000)

    mean_no_donate = float(jnp.mean(jnp.array(times_no_donate)))
    mean_donate = float(jnp.mean(jnp.array(times_donate)))

    print(f"M={M}, N={N}: no_donate={mean_no_donate:.3f}ms, donate={mean_donate:.3f}ms")


if __name__ == "__main__":
    test_jit_boundaries()
    test_gradient_checkpointing()
    test_vmap_batching()
    test_static_args()
    test_donate_argnums()

    print("\n" + "=" * 70)
    print("JAX OPTIMIZATION BENCHMARKS COMPLETE")
    print("=" * 70)
