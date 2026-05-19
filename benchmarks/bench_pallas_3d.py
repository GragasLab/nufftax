"""Benchmark + correctness for the 3D Pallas kernels (GPU).

Usage (on a CUDA host, e.g. the zay cluster):

    python benchmarks/bench_pallas_3d.py

It (1) verifies spread_3d_pallas / interp_3d_pallas match the pure-JAX
reference, and (2) times Pallas vs pure-JAX for spread, interp, and the
full nufft3d1/nufft3d2 forward + grad at several problem sizes.
"""

import time

import jax
import jax.numpy as jnp

from nufftax import nufft3d1
from nufftax.core.kernel import compute_kernel_params
from nufftax.core.spread import interp_3d_impl, spread_3d_impl


try:
    from nufftax.core.pallas_spread import interp_3d_pallas, spread_3d_pallas

    _HAS_PALLAS = True
except ImportError:
    _HAS_PALLAS = False


def check_gpu():
    devices = jax.devices()
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {devices}")
    return any(d.platform == "gpu" for d in devices)


def benchmark_fn(fn, *args, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - start) * 1000)
    return sorted(times)[len(times) // 2]


def rel_err(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-10))


def setup(M, nf, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    x = jax.random.uniform(k[0], (M,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32)
    y = jax.random.uniform(k[1], (M,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32)
    z = jax.random.uniform(k[2], (M,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32)
    c = (jax.random.normal(k[3], (M,)) + 1j * jax.random.normal(k[4], (M,))).astype(jnp.complex64)
    fw = (jax.random.normal(k[3], (nf, nf, nf)) + 1j * jax.random.normal(k[4], (nf, nf, nf))).astype(jnp.complex64)
    return x, y, z, c, fw


def main():
    if not check_gpu():
        print("\nNo GPU detected — the Pallas 3D kernels require a GPU. Aborting.")
        return
    if not _HAS_PALLAS:
        print("\nPallas import failed (need nufftax[cuda12]). Aborting.")
        return

    kp = compute_kernel_params(1e-6, upsampfac=2.0)

    print("\n=== Correctness (Pallas vs pure-JAX) ===")
    for M, nf in [(1024, 16), (50_000, 32), (200_000, 48)]:
        x, y, z, c, fw = setup(M, nf)
        s_p = spread_3d_pallas(x, y, z, c, nf, nf, nf, kp)
        s_r = spread_3d_impl(x, y, z, c, nf, nf, nf, kp)
        i_p = interp_3d_pallas(x, y, z, fw, kp)
        i_r = interp_3d_impl(x, y, z, fw, kp)
        print(f"  M={M:>8} nf={nf}: spread rel_err={rel_err(s_p, s_r):.2e}  interp rel_err={rel_err(i_p, i_r):.2e}")

    print("\n=== Speed: spread_3d (Pallas vs pure-JAX) ===")
    for M, nf in [(100_000, 48), (1_000_000, 64), (5_000_000, 64)]:
        x, y, z, c, _ = setup(M, nf)
        jfn = jax.jit(lambda x, y, z, c: spread_3d_impl(x, y, z, c, nf, nf, nf, kp))
        pfn = jax.jit(lambda x, y, z, c: spread_3d_pallas(x, y, z, c, nf, nf, nf, kp))
        tj = benchmark_fn(jfn, x, y, z, c)
        tp = benchmark_fn(pfn, x, y, z, c)
        print(f"  M={M:>8} nf={nf}^3: JAX={tj:8.2f}ms  Pallas={tp:8.2f}ms  speedup={tj / tp:.2f}x")

    print("\n=== Speed: interp_3d (Pallas vs pure-JAX) ===")
    for M, nf in [(100_000, 48), (1_000_000, 64), (5_000_000, 64)]:
        x, y, z, _, fw = setup(M, nf)
        jfn = jax.jit(lambda x, y, z, fw: interp_3d_impl(x, y, z, fw, kp))
        pfn = jax.jit(lambda x, y, z, fw: interp_3d_pallas(x, y, z, fw, kp))
        tj = benchmark_fn(jfn, x, y, z, fw)
        tp = benchmark_fn(pfn, x, y, z, fw)
        print(f"  M={M:>8} nf={nf}^3: JAX={tj:8.2f}ms  Pallas={tp:8.2f}ms  speedup={tj / tp:.2f}x")

    print("\n=== End-to-end nufft3d1 forward + grad (dispatch picks Pallas above thresholds) ===")
    for M, N in [(100_000, 48), (1_000_000, 64)]:
        x, y, z, c, _ = setup(M, N)
        fwd = jax.jit(lambda x, y, z, c: nufft3d1(x, y, z, c, (N, N, N), eps=1e-6))
        grad = jax.jit(jax.grad(lambda c: jnp.sum(jnp.abs(nufft3d1(x, y, z, c, (N, N, N), eps=1e-6)) ** 2).real))
        tf = benchmark_fn(fwd, x, y, z, c)
        tg = benchmark_fn(grad, c)
        print(f"  M={M:>8} N={N}^3: forward={tf:8.2f}ms  grad={tg:8.2f}ms")


if __name__ == "__main__":
    main()
