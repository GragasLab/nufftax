"""
Comprehensive NUFFT benchmark for realistic shapes.

This benchmark measures:
1. Forward pass (spread/interp)
2. Backward pass (gradients)
3. Memory usage
4. Comparison of fused vs separate kernel+derivative
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Import NUFFT components
from nufftax.core.kernel import (
    compute_kernel_params,
    es_kernel,
    es_kernel_derivative,
    es_kernel_with_derivative,
)
from nufftax.core.spread import (
    spread_1d,
    spread_2d,
    spread_3d,
    interp_1d,
    interp_2d,
    interp_3d,
)
from nufftax.transforms import nufft1d1


def format_bytes(size_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def benchmark_kernel_functions(M=1_000_000, nspread=8, n_runs=100):
    """Benchmark kernel evaluation: separate vs fused."""
    print("\n" + "="*60)
    print("KERNEL FUNCTION BENCHMARK")
    print("="*60)
    print(f"M = {M:,} points, nspread = {nspread}, n_runs = {n_runs}")

    beta = 12.0
    c = 4.0 / (nspread ** 2)

    key = jax.random.PRNGKey(42)
    z = jax.random.uniform(key, (M,), minval=-nspread/2, maxval=nspread/2)

    # JIT compile both approaches
    @jax.jit
    def separate_kernel_derivative(z, beta, c):
        phi = es_kernel(z, beta, c)
        dphi = es_kernel_derivative(z, beta, c)
        return phi, dphi

    @jax.jit
    def fused_kernel_derivative(z, beta, c):
        return es_kernel_with_derivative(z, beta, c)

    # Warm up
    _ = separate_kernel_derivative(z, beta, c)
    _ = fused_kernel_derivative(z, beta, c)
    jax.block_until_ready(_)

    # Benchmark separate
    start = time.perf_counter()
    for _ in range(n_runs):
        result = separate_kernel_derivative(z, beta, c)
        jax.block_until_ready(result)
    elapsed_separate = (time.perf_counter() - start) / n_runs * 1000

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(n_runs):
        result = fused_kernel_derivative(z, beta, c)
        jax.block_until_ready(result)
    elapsed_fused = (time.perf_counter() - start) / n_runs * 1000

    print(f"\nSeparate kernel + derivative: {elapsed_separate:.3f} ms")
    print(f"Fused kernel + derivative:    {elapsed_fused:.3f} ms")
    speedup = elapsed_separate / elapsed_fused
    print(f"Speedup: {speedup:.2f}x")

    return {"separate_ms": elapsed_separate, "fused_ms": elapsed_fused, "speedup": speedup}


def benchmark_spread_interp_1d(M=100_000, N=256, tol=1e-6, n_runs=50):
    """Benchmark 1D spread and interpolation."""
    print("\n" + "="*60)
    print("1D SPREAD/INTERPOLATION BENCHMARK")
    print("="*60)
    print(f"M = {M:,} points, N = {N}, tol = {tol}")

    kernel_params = compute_kernel_params(tol)
    nf = int(N * kernel_params.upsampfac)

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(key, (M,), dtype=jnp.complex64)
    fw = jax.random.normal(jax.random.PRNGKey(43), (nf,), dtype=jnp.complex64)

    # JIT compile
    spread_jit = jax.jit(partial(spread_1d, nf=nf, kernel_params=kernel_params))
    interp_jit = jax.jit(partial(interp_1d, nf=nf, kernel_params=kernel_params))

    # Warm up
    _ = spread_jit(x, c)
    _ = interp_jit(x, fw)
    jax.block_until_ready(_)

    # Benchmark spread
    start = time.perf_counter()
    for _ in range(n_runs):
        result = spread_jit(x, c)
        jax.block_until_ready(result)
    spread_time = (time.perf_counter() - start) / n_runs * 1000

    # Benchmark interp
    start = time.perf_counter()
    for _ in range(n_runs):
        result = interp_jit(x, fw)
        jax.block_until_ready(result)
    interp_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"\nSpread 1D: {spread_time:.3f} ms")
    print(f"Interp 1D: {interp_time:.3f} ms")

    # Benchmark gradients
    def loss_spread(x, c):
        return jnp.sum(jnp.abs(spread_jit(x, c))**2)

    def loss_interp(x, fw):
        return jnp.sum(jnp.abs(interp_jit(x, fw))**2)

    grad_spread = jax.jit(jax.grad(loss_spread, argnums=(0, 1)))
    grad_interp = jax.jit(jax.grad(loss_interp, argnums=(0, 1)))

    # Warm up grads
    _ = grad_spread(x, c)
    _ = grad_interp(x, fw)
    jax.block_until_ready(_)

    # Benchmark grad spread
    start = time.perf_counter()
    for _ in range(n_runs):
        result = grad_spread(x, c)
        jax.block_until_ready(result)
    grad_spread_time = (time.perf_counter() - start) / n_runs * 1000

    # Benchmark grad interp
    start = time.perf_counter()
    for _ in range(n_runs):
        result = grad_interp(x, fw)
        jax.block_until_ready(result)
    grad_interp_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"\nGrad Spread 1D: {grad_spread_time:.3f} ms")
    print(f"Grad Interp 1D: {grad_interp_time:.3f} ms")

    return {
        "spread_ms": spread_time,
        "interp_ms": interp_time,
        "grad_spread_ms": grad_spread_time,
        "grad_interp_ms": grad_interp_time,
    }


def benchmark_spread_interp_2d(M=100_000, N1=128, N2=128, tol=1e-6, n_runs=30):
    """Benchmark 2D spread and interpolation."""
    print("\n" + "="*60)
    print("2D SPREAD/INTERPOLATION BENCHMARK")
    print("="*60)
    print(f"M = {M:,} points, N = ({N1}, {N2}), tol = {tol}")

    kernel_params = compute_kernel_params(tol)
    nf1 = int(N1 * kernel_params.upsampfac)
    nf2 = int(N2 * kernel_params.upsampfac)

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
    y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(jax.random.PRNGKey(44), (M,), dtype=jnp.complex64)
    fw = jax.random.normal(jax.random.PRNGKey(45), (nf1, nf2), dtype=jnp.complex64)

    # JIT compile
    spread_jit = jax.jit(partial(spread_2d, nf1=nf1, nf2=nf2, kernel_params=kernel_params))
    interp_jit = jax.jit(partial(interp_2d, nf1=nf1, nf2=nf2, kernel_params=kernel_params))

    # Warm up
    _ = spread_jit(x, y, c)
    _ = interp_jit(x, y, fw)
    jax.block_until_ready(_)

    # Benchmark spread
    start = time.perf_counter()
    for _ in range(n_runs):
        result = spread_jit(x, y, c)
        jax.block_until_ready(result)
    spread_time = (time.perf_counter() - start) / n_runs * 1000

    # Benchmark interp
    start = time.perf_counter()
    for _ in range(n_runs):
        result = interp_jit(x, y, fw)
        jax.block_until_ready(result)
    interp_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"\nSpread 2D: {spread_time:.3f} ms")
    print(f"Interp 2D: {interp_time:.3f} ms")

    # Benchmark gradients
    def loss_spread(x, y, c):
        return jnp.sum(jnp.abs(spread_jit(x, y, c))**2)

    grad_spread = jax.jit(jax.grad(loss_spread, argnums=(0, 1, 2)))

    # Warm up grads
    _ = grad_spread(x, y, c)
    jax.block_until_ready(_)

    # Benchmark grad spread
    start = time.perf_counter()
    for _ in range(n_runs):
        result = grad_spread(x, y, c)
        jax.block_until_ready(result)
    grad_spread_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"\nGrad Spread 2D: {grad_spread_time:.3f} ms")

    return {
        "spread_ms": spread_time,
        "interp_ms": interp_time,
        "grad_spread_ms": grad_spread_time,
    }


def benchmark_nufft1_end_to_end(M=100_000, N=256, tol=1e-6, n_runs=30):
    """Benchmark end-to-end NUFFT type 1."""
    print("\n" + "="*60)
    print("END-TO-END NUFFT TYPE 1 (1D) BENCHMARK")
    print("="*60)
    print(f"M = {M:,} points, N = {N}, tol = {tol}")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(jax.random.PRNGKey(43), (M,), dtype=jnp.complex64)

    # JIT compile with static_argnums for output_shape and tol
    @jax.jit
    def nufft1_wrapper(x, c):
        return nufft1d1(x, c, n_modes=N, eps=tol)

    # Warm up
    _ = nufft1_wrapper(x, c)
    jax.block_until_ready(_)

    # Benchmark forward
    start = time.perf_counter()
    for _ in range(n_runs):
        result = nufft1_wrapper(x, c)
        jax.block_until_ready(result)
    forward_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"\nNUFFT1 forward: {forward_time:.3f} ms")

    # Benchmark gradient
    def loss(x, c):
        return jnp.sum(jnp.abs(nufft1_wrapper(x, c))**2)

    grad_nufft = jax.jit(jax.grad(loss, argnums=(0, 1)))

    # Warm up
    _ = grad_nufft(x, c)
    jax.block_until_ready(_)

    # Benchmark gradient
    start = time.perf_counter()
    for _ in range(n_runs):
        result = grad_nufft(x, c)
        jax.block_until_ready(result)
    grad_time = (time.perf_counter() - start) / n_runs * 1000

    print(f"NUFFT1 gradient: {grad_time:.3f} ms")

    return {"forward_ms": forward_time, "gradient_ms": grad_time}


def benchmark_memory_usage():
    """Estimate memory usage for different problem sizes."""
    print("\n" + "="*60)
    print("MEMORY USAGE ANALYSIS")
    print("="*60)

    # Analyze memory for different M and N
    for M in [10_000, 100_000, 1_000_000]:
        for N in [64, 256, 1024]:
            kernel_params = compute_kernel_params(1e-6)
            nf = int(N * kernel_params.upsampfac)
            nspread = kernel_params.nspread

            # Memory for inputs
            input_mem = M * 8  # x coordinates (float64)
            input_mem += M * 8  # c values (complex64 = 8 bytes)

            # Memory for indices and weights
            indices_mem = M * nspread * 4  # int32 indices
            weights_mem = M * nspread * 4  # float32 weights

            # Memory for output grid
            output_mem = nf * 8  # complex64

            # During gradient: need to store intermediate values
            grad_mem = indices_mem + 2 * weights_mem  # weights + dweights

            total_fwd = input_mem + indices_mem + weights_mem + output_mem
            total_grad = total_fwd + grad_mem

            print(f"\nM={M:>10,}, N={N:>4}: "
                  f"Forward={format_bytes(total_fwd):>10}, "
                  f"Gradient={format_bytes(total_grad):>10}")


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "="*60)
    print("NUFFTAX COMPREHENSIVE BENCHMARK")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"Device: {jax.devices()[0]}")

    results = {}

    # Kernel benchmarks
    results["kernel"] = benchmark_kernel_functions(M=1_000_000, n_runs=100)

    # 1D benchmarks
    results["1d"] = benchmark_spread_interp_1d(M=100_000, N=256, n_runs=50)

    # 2D benchmarks
    results["2d"] = benchmark_spread_interp_2d(M=100_000, N1=128, N2=128, n_runs=30)

    # End-to-end benchmarks
    results["nufft1"] = benchmark_nufft1_end_to_end(M=100_000, N=256, n_runs=30)

    # Memory analysis
    benchmark_memory_usage()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nKernel fusion speedup: {results['kernel']['speedup']:.2f}x")
    print(f"\n1D Operations (M=100k, N=256):")
    print(f"  Spread: {results['1d']['spread_ms']:.3f} ms")
    print(f"  Interp: {results['1d']['interp_ms']:.3f} ms")
    print(f"  Grad Spread: {results['1d']['grad_spread_ms']:.3f} ms")
    print(f"\n2D Operations (M=100k, N=128x128):")
    print(f"  Spread: {results['2d']['spread_ms']:.3f} ms")
    print(f"  Interp: {results['2d']['interp_ms']:.3f} ms")
    print(f"\nEnd-to-end NUFFT1 (M=100k, N=256):")
    print(f"  Forward: {results['nufft1']['forward_ms']:.3f} ms")
    print(f"  Gradient: {results['nufft1']['gradient_ms']:.3f} ms")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
