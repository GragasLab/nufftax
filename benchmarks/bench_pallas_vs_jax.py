"""NUFFTAX GPU benchmark.

Two things, streamed:
  1. spread Pallas kernel vs pure-JAX `spread_*_impl` (1D/2D/3D) -- the fused
     Triton scatter is the speedup that matters.
  2. end-to-end public NUFFT (Type 1 & 2) on the default path (Pallas spread
     auto-dispatched on GPU above the per-dim _PALLAS_MIN_M_SPREAD_* thresholds).

Interpolation (Type 2 gather) has no Pallas kernel by design: XLA already
fuses it optimally (benchmarked: parity in 1D/2D, 2-4x slower in 3D).

Usage:  python benchmarks/bench_pallas_vs_jax.py
"""

import os
import sys
import time


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp

from nufftax import nufft1d1, nufft1d2, nufft2d1, nufft2d2, nufft3d1, nufft3d2
from nufftax.core.kernel import compute_kernel_params
from nufftax.core.spread import spread_1d_impl, spread_2d_impl, spread_3d_impl


try:
    from nufftax.core.pallas_spread import spread_1d_pallas, spread_2d_pallas, spread_3d_pallas

    HAS_PALLAS = True
except ImportError:
    HAS_PALLAS = False


def log(*a):
    print(*a, flush=True)


def bench(fn, *a, n=15):
    for _ in range(3):
        jax.block_until_ready(fn(*a))
    t = []
    for _ in range(n):
        s = time.perf_counter()
        jax.block_until_ready(fn(*a))
        t.append((time.perf_counter() - s) * 1000)
    return sorted(t)[len(t) // 2]


def main():
    log("device:", jax.devices())
    if not HAS_PALLAS:
        log("Pallas unavailable; spread comparison skipped.")
    kp = compute_kernel_params(1e-6, 2.0)
    K = jax.random.PRNGKey(0)

    if HAS_PALLAS:
        log("\n== spread: pure-JAX impl vs Pallas ==")
        for M in [100_000, 1_000_000]:
            x = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
            y = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
            z = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
            c = jax.random.normal(K, (M,), dtype=jnp.complex64)
            r1 = bench(jax.jit(lambda x, c: spread_1d_impl(x, c, 512, kp)), x, c)
            p1 = bench(jax.jit(lambda x, c: spread_1d_pallas(x, c, 512, kp)), x, c)
            r2 = bench(jax.jit(lambda x, y, c: spread_2d_impl(x, y, c, 128, 128, kp)), x, y, c)
            p2 = bench(jax.jit(lambda x, y, c: spread_2d_pallas(x, y, c, 128, 128, kp)), x, y, c)
            r3 = bench(jax.jit(lambda x, y, z, c: spread_3d_impl(x, y, z, c, 128, 128, 128, kp)), x, y, z, c)
            p3 = bench(jax.jit(lambda x, y, z, c: spread_3d_pallas(x, y, z, c, 128, 128, 128, kp)), x, y, z, c)
            log(f"  M={M:>9} 1D impl={r1:7.2f} pallas={p1:7.2f} ({r1 / p1:5.1f}x)"
                f"  2D impl={r2:7.2f} pallas={p2:7.2f} ({r2 / p2:4.1f}x)"
                f"  3D impl={r3:7.2f} pallas={p3:7.2f} ({r3 / p3:4.1f}x)")

    log("\n== end-to-end public NUFFT (default path) ==")
    for M in [100_000, 1_000_000, 10_000_000]:
        x = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(1), (M,), minval=-jnp.pi, maxval=jnp.pi)
        z = jax.random.uniform(jax.random.PRNGKey(2), (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(K, (M,), dtype=jnp.complex64)
        t1 = bench(jax.jit(lambda x, c: nufft1d1(x, c, 256, eps=1e-6)), x, c)
        t2 = bench(jax.jit(lambda x, y, c: nufft2d1(x, y, c, (256, 256), eps=1e-6)), x, y, c)
        t3 = bench(jax.jit(lambda x, y, z, c: nufft3d1(x, y, z, c, (128, 128, 128), eps=1e-6)), x, y, z, c)
        log(f"  M={M:>9} Type1  1d1(256)={t1:7.2f}ms  2d1(256^2)={t2:7.2f}ms  3d1(128^3)={t3:7.2f}ms")

    N = 256
    for M in [100_000, 1_000_000]:
        x = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(1), (M,), minval=-jnp.pi, maxval=jnp.pi)
        z = jax.random.uniform(jax.random.PRNGKey(2), (M,), minval=-jnp.pi, maxval=jnp.pi)
        f1 = jax.random.normal(K, (N,), dtype=jnp.complex64)
        f2 = jax.random.normal(K, (N, N), dtype=jnp.complex64)
        f3 = jax.random.normal(K, (128, 128, 128), dtype=jnp.complex64)
        t1 = bench(jax.jit(lambda x, f: nufft1d2(x, f, eps=1e-6)), x, f1)
        t2 = bench(jax.jit(lambda x, y, f: nufft2d2(x, y, f, eps=1e-6)), x, y, f2)
        t3 = bench(jax.jit(lambda x, y, z, f: nufft3d2(x, y, z, f, eps=1e-6)), x, y, z, f3)
        log(f"  M={M:>9} Type2  1d2(256)={t1:7.2f}ms  2d2(256^2)={t2:7.2f}ms  3d2(128^3)={t3:7.2f}ms")
    log("BENCHEND")


if __name__ == "__main__":
    main()
