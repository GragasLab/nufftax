"""Threshold probe: at what M does Pallas spread start beating pure-JAX impl?
Sweeps small M (100..100K) for 1D/2D/3D. Streams output."""
import sys
import time

import jax
import jax.numpy as jnp

from nufftax.core.kernel import compute_kernel_params
from nufftax.core.pallas_spread import spread_1d_pallas, spread_2d_pallas, spread_3d_pallas
from nufftax.core.spread import spread_1d_impl, spread_2d_impl, spread_3d_impl


def log(*a):
    print(*a, flush=True)

def b(f, *a, n=30):
    for _ in range(5):
        jax.block_until_ready(f(*a))
    t = []
    for _ in range(n):
        s = time.perf_counter()
        jax.block_until_ready(f(*a))
        t.append((time.perf_counter()-s)*1000)
    return sorted(t)[len(t)//2]

log("device:", jax.devices())
kp = compute_kernel_params(1e-6, 2.0)
K = jax.random.PRNGKey(0)
Ms = [100, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000]

log("\n== 1D (nf=512) ==")
nf = 512
for M in Ms:
    x = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(K, (M,), dtype=jnp.complex64)
    fi = jax.jit(lambda x,c: spread_1d_impl(x, c, nf, kp))
    fp = jax.jit(lambda x,c: spread_1d_pallas(x, c, nf, kp))
    ti = b(fi, x, c)
    tp = b(fp, x, c)
    log(f"  M={M:>7} impl={ti:7.3f}ms pallas={tp:7.3f}ms speedup={ti/tp:6.2f}x  {'pallas-WIN' if tp<ti else 'impl-WIN'}")

log("\n== 2D (nf=128) ==")
nf = 128
for M in Ms:
    x = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
    y = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(K, (M,), dtype=jnp.complex64)
    fi = jax.jit(lambda x,y,c: spread_2d_impl(x, y, c, nf, nf, kp))
    fp = jax.jit(lambda x,y,c: spread_2d_pallas(x, y, c, nf, nf, kp))
    ti = b(fi, x, y, c)
    tp = b(fp, x, y, c)
    log(f"  M={M:>7} impl={ti:7.3f}ms pallas={tp:7.3f}ms speedup={ti/tp:6.2f}x  {'pallas-WIN' if tp<ti else 'impl-WIN'}")

log("\n== 3D (nf=64) ==")
nf = 64
for M in Ms:
    x = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
    y = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
    z = jax.random.uniform(K, (M,), minval=-jnp.pi, maxval=jnp.pi)
    c = jax.random.normal(K, (M,), dtype=jnp.complex64)
    fi = jax.jit(lambda x,y,z,c: spread_3d_impl(x, y, z, c, nf, nf, nf, kp))
    fp = jax.jit(lambda x,y,z,c: spread_3d_pallas(x, y, z, c, nf, nf, nf, kp))
    ti = b(fi, x, y, z, c)
    tp = b(fp, x, y, z, c)
    log(f"  M={M:>7} impl={ti:7.3f}ms pallas={tp:7.3f}ms speedup={ti/tp:6.2f}x  {'pallas-WIN' if tp<ti else 'impl-WIN'}")
log("THDONE")
sys.stdout.flush()
