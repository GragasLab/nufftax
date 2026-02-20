"""
Benchmark: Pure JAX vs Pallas GPU kernels for NUFFT.

Compares spread/interp primitives (1D, 2D) and end-to-end NUFFT Type 1 & 2.

Usage:
    python benchmarks/bench_pallas_vs_jax.py
"""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from functools import partial

import jax
import jax.numpy as jnp

from nufftax.core.deconvolve import deconvolve_shuffle_1d, deconvolve_shuffle_2d
from nufftax.core.kernel import compute_kernel_params, kernel_fourier_series
from nufftax.core.spread import interp_1d_impl, interp_2d_impl, spread_1d_impl, spread_2d_impl
from nufftax.transforms.nufft1 import nufft1d1, nufft2d1
from nufftax.transforms.nufft2 import nufft1d2, nufft2d2
from nufftax.utils.grid import compute_grid_size


try:
    from nufftax.core.pallas_spread import (
        interp_1d_pallas,
        interp_2d_pallas,
        spread_1d_pallas,
        spread_2d_pallas,
    )

    HAS_PALLAS = True
except ImportError as e:
    print(f"WARNING: Pallas not available: {e}")
    HAS_PALLAS = False


def _nufft1d1_pallas(x, c, n_modes, eps=1e-6, isign=1, upsampfac=2.0, modeord=0):
    kernel_params = compute_kernel_params(eps, upsampfac)
    nf = compute_grid_size(n_modes, upsampfac, kernel_params.nspread)
    dtype = jnp.real(c).dtype
    phihat = kernel_fourier_series(nf, kernel_params.nspread, kernel_params.beta, kernel_params.c, dtype=dtype)
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    fw = spread_1d_pallas(x_normalized, c, nf, kernel_params)[None, :]
    fw_hat = jnp.fft.ifft(fw, axis=-1) * nf if isign > 0 else jnp.fft.fft(fw, axis=-1)
    return deconvolve_shuffle_1d(fw_hat, phihat, n_modes, modeord)[0]


def _nufft2d1_pallas(x, y, c, n_modes, eps=1e-6, isign=1, upsampfac=2.0, modeord=0):
    if isinstance(n_modes, int):
        n_modes1 = n_modes2 = n_modes
    else:
        n_modes1, n_modes2 = n_modes
    kernel_params = compute_kernel_params(eps, upsampfac)
    nf1 = compute_grid_size(n_modes1, upsampfac, kernel_params.nspread)
    nf2 = compute_grid_size(n_modes2, upsampfac, kernel_params.nspread)
    dtype = jnp.real(c).dtype
    phihat1 = kernel_fourier_series(nf1, kernel_params.nspread, kernel_params.beta, kernel_params.c, dtype=dtype)
    phihat2 = kernel_fourier_series(nf2, kernel_params.nspread, kernel_params.beta, kernel_params.c, dtype=dtype)
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    y_normalized = jnp.mod(y + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    fw = spread_2d_pallas(x_normalized, y_normalized, c, nf1, nf2, kernel_params)[None, :, :]
    fw_hat = jnp.fft.ifft2(fw, axes=(-2, -1)) * (nf1 * nf2) if isign > 0 else jnp.fft.fft2(fw, axes=(-2, -1))
    return deconvolve_shuffle_2d(fw_hat, phihat1, phihat2, n_modes1, n_modes2, modeord)[0]


def check_gpu():
    devices = jax.devices()
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {devices}")
    for d in devices:
        if d.platform == "gpu":
            print(f"GPU: {d.device_kind}")
    return any(d.platform == "gpu" for d in devices)


def benchmark_fn(fn, *args, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        result = fn(*args)
        jax.block_until_ready(result)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    times_ms = [t * 1000 for t in times]
    median = sorted(times_ms)[len(times_ms) // 2]
    return median, result


def check_correctness(result_jax, result_pallas, label, rtol=1e-3):
    err = jnp.max(jnp.abs(result_jax - result_pallas))
    rel = err / (jnp.max(jnp.abs(result_jax)) + 1e-10)
    ok = float(rel) < rtol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: rel_err={float(rel):.2e}")
    return ok


def print_row(label, t_jax, t_pallas):
    speedup = t_jax / t_pallas
    print(f"  {label}: JAX={t_jax:7.3f}ms  Pallas={t_pallas:7.3f}ms  speedup={speedup:.1f}x")
    return {"jax_ms": t_jax, "pallas_ms": t_pallas, "speedup": speedup}


def bench_spread_1d(M, N, tol=1e-6, n_runs=30):
    kernel_params = compute_kernel_params(tol)
    nf = int(N * kernel_params.upsampfac)
    x = jax.random.uniform(jax.random.PRNGKey(42), (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(jax.random.PRNGKey(43), (M,), dtype=jnp.complex64)
    jax_fn = jax.jit(partial(spread_1d_impl, nf=nf, kernel_params=kernel_params))
    pallas_fn = jax.jit(partial(spread_1d_pallas, nf=nf, kernel_params=kernel_params))
    t_jax, res_jax = benchmark_fn(jax_fn, x, c, n_runs=n_runs)
    t_pallas, res_pallas = benchmark_fn(pallas_fn, x, c, n_runs=n_runs)
    r = print_row(f"Spread 1D (M={M:>8,}, N={N:>4})", t_jax, t_pallas)
    check_correctness(res_jax, res_pallas.astype(res_jax.dtype), "spread_1d")
    return r


def bench_spread_2d(M, N1, N2, tol=1e-6, n_runs=20):
    kernel_params = compute_kernel_params(tol)
    nf1 = int(N1 * kernel_params.upsampfac)
    nf2 = int(N2 * kernel_params.upsampfac)
    x = jax.random.uniform(jax.random.PRNGKey(42), (M,), minval=-jnp.pi, maxval=jnp.pi)
    y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(jax.random.PRNGKey(44), (M,), dtype=jnp.complex64)
    jax_fn = jax.jit(partial(spread_2d_impl, nf1=nf1, nf2=nf2, kernel_params=kernel_params))
    pallas_fn = jax.jit(partial(spread_2d_pallas, nf1=nf1, nf2=nf2, kernel_params=kernel_params))
    t_jax, res_jax = benchmark_fn(jax_fn, x, y, c, n_runs=n_runs)
    t_pallas, res_pallas = benchmark_fn(pallas_fn, x, y, c, n_runs=n_runs)
    r = print_row(f"Spread 2D (M={M:>8,}, N={N1}x{N2})", t_jax, t_pallas)
    check_correctness(res_jax, res_pallas.astype(res_jax.dtype), "spread_2d")
    return r


def bench_nufft1d1(M, N, tol=1e-6, n_runs=20):
    x = jax.random.uniform(jax.random.PRNGKey(42), (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(jax.random.PRNGKey(43), (M,), dtype=jnp.complex64)

    @jax.jit
    def jax_fn(x, c):
        return nufft1d1(x, c, n_modes=N, eps=tol)

    @jax.jit
    def pallas_fn(x, c):
        return _nufft1d1_pallas(x, c, n_modes=N, eps=tol)

    t_jax, res_jax = benchmark_fn(jax_fn, x, c, n_runs=n_runs)
    t_pallas, res_pallas = benchmark_fn(pallas_fn, x, c, n_runs=n_runs)
    r = print_row(f"NUFFT1D1  (M={M:>8,}, N={N:>4})", t_jax, t_pallas)
    check_correctness(res_jax, res_pallas, "nufft1d1")
    return r


def bench_nufft2d1(M, N1, N2, tol=1e-6, n_runs=20):
    x = jax.random.uniform(jax.random.PRNGKey(42), (M,), minval=-jnp.pi, maxval=jnp.pi)
    y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(jax.random.PRNGKey(44), (M,), dtype=jnp.complex64)

    @jax.jit
    def jax_fn(x, y, c):
        return nufft2d1(x, y, c, n_modes=(N1, N2), eps=tol)

    @jax.jit
    def pallas_fn(x, y, c):
        return _nufft2d1_pallas(x, y, c, n_modes=(N1, N2), eps=tol)

    t_jax, res_jax = benchmark_fn(jax_fn, x, y, c, n_runs=n_runs)
    t_pallas, res_pallas = benchmark_fn(pallas_fn, x, y, c, n_runs=n_runs)
    r = print_row(f"NUFFT2D1  (M={M:>8,}, N={N1}x{N2})", t_jax, t_pallas)
    check_correctness(res_jax, res_pallas, "nufft2d1")
    return r


def bench_interp_1d(M, N, tol=1e-6, n_runs=30):
    kernel_params = compute_kernel_params(tol)
    nf = int(N * kernel_params.upsampfac)
    x = jax.random.uniform(jax.random.PRNGKey(42), (M,), minval=-jnp.pi, maxval=jnp.pi)
    fw = jax.random.normal(jax.random.PRNGKey(43), (nf,), dtype=jnp.complex64)
    jax_fn = jax.jit(partial(interp_1d_impl, kernel_params=kernel_params))
    pallas_fn = jax.jit(partial(interp_1d_pallas, kernel_params=kernel_params))
    t_jax, res_jax = benchmark_fn(jax_fn, x, fw, n_runs=n_runs)
    t_pallas, res_pallas = benchmark_fn(pallas_fn, x, fw, n_runs=n_runs)
    r = print_row(f"Interp 1D (M={M:>8,}, N={N:>4})", t_jax, t_pallas)
    check_correctness(res_jax, res_pallas.astype(res_jax.dtype), "interp_1d")
    return r


def bench_interp_2d(M, N1, N2, tol=1e-6, n_runs=20):
    kernel_params = compute_kernel_params(tol)
    nf1 = int(N1 * kernel_params.upsampfac)
    nf2 = int(N2 * kernel_params.upsampfac)
    x = jax.random.uniform(jax.random.PRNGKey(42), (M,), minval=-jnp.pi, maxval=jnp.pi)
    y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
    fw = jax.random.normal(jax.random.PRNGKey(44), (nf2, nf1), dtype=jnp.complex64)
    jax_fn = jax.jit(partial(interp_2d_impl, kernel_params=kernel_params))
    pallas_fn = jax.jit(partial(interp_2d_pallas, kernel_params=kernel_params))
    t_jax, res_jax = benchmark_fn(jax_fn, x, y, fw, n_runs=n_runs)
    t_pallas, res_pallas = benchmark_fn(pallas_fn, x, y, fw, n_runs=n_runs)
    r = print_row(f"Interp 2D (M={M:>8,}, N={N1}x{N2})", t_jax, t_pallas)
    check_correctness(res_jax, res_pallas.astype(res_jax.dtype), "interp_2d")
    return r


def bench_nufft1d2(M, N, tol=1e-6, n_runs=20):
    x = jax.random.uniform(jax.random.PRNGKey(42), (M,), minval=-jnp.pi, maxval=jnp.pi)
    f = jax.random.normal(jax.random.PRNGKey(43), (N,), dtype=jnp.complex64)

    @jax.jit
    def fn(x, f):
        return nufft1d2(x, f, eps=tol)

    t, res = benchmark_fn(fn, x, f, n_runs=n_runs)
    print(f"  NUFFT1D2  (M={M:>8,}, N={N:>4}): {t:7.3f}ms")
    return {"ms": t}


def bench_nufft2d2(M, N1, N2, tol=1e-6, n_runs=20):
    x = jax.random.uniform(jax.random.PRNGKey(42), (M,), minval=-jnp.pi, maxval=jnp.pi)
    y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
    f = jax.random.normal(jax.random.PRNGKey(44), (N2, N1), dtype=jnp.complex64)

    @jax.jit
    def fn(x, y, f):
        return nufft2d2(x, y, f, eps=tol)

    t, res = benchmark_fn(fn, x, y, f, n_runs=n_runs)
    print(f"  NUFFT2D2  (M={M:>8,}, N={N1}x{N2}): {t:7.3f}ms")
    return {"ms": t}


def run_benchmarks():
    print("=" * 75)
    print("NUFFTAX: Pure JAX vs Pallas (Triton) GPU Kernels")
    print("=" * 75)

    is_gpu = check_gpu()
    if not HAS_PALLAS:
        print("\nPallas kernels not available. Exiting.")
        sys.exit(1)
    if not is_gpu:
        print("\nNo GPU available. Exiting.")
        sys.exit(1)

    results = {}

    print("\n" + "-" * 75)
    print("1D SPREADING (varying M, N=256)")
    print("-" * 75)
    for M in [10_000, 100_000, 1_000_000, 10_000_000]:
        try:
            results[f"spread_1d_M{M}"] = bench_spread_1d(M, N=256)
        except Exception as e:
            print(f"  ERROR M={M}: {e}")

    print("\n" + "-" * 75)
    print("2D SPREADING (varying M, N=64x64)")
    print("-" * 75)
    for M in [10_000, 100_000, 1_000_000]:
        try:
            results[f"spread_2d_M{M}"] = bench_spread_2d(M, N1=64, N2=64)
        except Exception as e:
            print(f"  ERROR M={M}: {e}")

    print("\n" + "-" * 75)
    print("1D INTERPOLATION (varying M, N=256)")
    print("-" * 75)
    for M in [10_000, 100_000, 1_000_000, 10_000_000]:
        try:
            results[f"interp_1d_M{M}"] = bench_interp_1d(M, N=256)
        except Exception as e:
            print(f"  ERROR M={M}: {e}")

    print("\n" + "-" * 75)
    print("2D INTERPOLATION (varying M, N=64x64)")
    print("-" * 75)
    for M in [10_000, 100_000, 1_000_000]:
        try:
            results[f"interp_2d_M{M}"] = bench_interp_2d(M, N1=64, N2=64)
        except Exception as e:
            print(f"  ERROR M={M}: {e}")

    print("\n" + "-" * 75)
    print("END-TO-END NUFFT TYPE 1 - 1D (varying M, N=256)")
    print("-" * 75)
    for M in [10_000, 100_000, 1_000_000]:
        try:
            results[f"nufft1d1_M{M}"] = bench_nufft1d1(M, N=256)
        except Exception as e:
            print(f"  ERROR M={M}: {e}")

    print("\n" + "-" * 75)
    print("END-TO-END NUFFT TYPE 1 - 2D (varying M, N=64x64)")
    print("-" * 75)
    for M in [10_000, 100_000, 1_000_000]:
        try:
            results[f"nufft2d1_M{M}"] = bench_nufft2d1(M, N1=64, N2=64)
        except Exception as e:
            print(f"  ERROR M={M}: {e}")

    print("\n" + "-" * 75)
    print("END-TO-END NUFFT TYPE 2 - 1D (auto-dispatch, varying M, N=256)")
    print("-" * 75)
    for M in [10_000, 100_000, 1_000_000]:
        try:
            bench_nufft1d2(M, N=256)
        except Exception as e:
            print(f"  ERROR M={M}: {e}")

    print("\n" + "-" * 75)
    print("END-TO-END NUFFT TYPE 2 - 2D (auto-dispatch, varying M, N=64x64)")
    print("-" * 75)
    for M in [10_000, 100_000, 1_000_000]:
        try:
            bench_nufft2d2(M, N1=64, N2=64)
        except Exception as e:
            print(f"  ERROR M={M}: {e}")

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"\n{'Test':<40} {'JAX (ms)':>10} {'Pallas (ms)':>12} {'Speedup':>10}")
    print("-" * 75)
    for name, r in results.items():
        if "jax_ms" in r:
            print(f"  {name:<38} {r['jax_ms']:>10.3f} {r['pallas_ms']:>12.3f} {r['speedup']:>9.1f}x")

    speedups = [r["speedup"] for r in results.values() if "speedup" in r]
    if speedups:
        geo = 1.0
        for s in speedups:
            geo *= s
        geo = geo ** (1.0 / len(speedups))
        print(f"\n  Geometric mean speedup: {geo:.1f}x")


if __name__ == "__main__":
    run_benchmarks()
