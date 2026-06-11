"""GPU-only: reverse-mode gradients w.r.t. nonuniform positions, through Pallas.

Covers the case raised on PR #11 — reverse-mode ``grad`` w.r.t. the nonuniform
positions of a Type 1 transform (e.g. ``nufft2d1``) while the fused Pallas
spreading backend is active. The old size-threshold dispatch never routed the
small-M gradient tests through Pallas, so this path was previously untested.

Skipped unless a GPU is present and the Pallas backend is enabled
(``NUFFTAX_PALLAS_BACKEND``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nufftax import nufft2d1
from nufftax.core.kernel import compute_kernel_params
from nufftax.core.spread import _HAS_PALLAS_GPU, _USE_PALLAS_SPREAD, spread_2d


pallas_gpu_only = pytest.mark.skipif(
    not (_HAS_PALLAS_GPU and _USE_PALLAS_SPREAD),
    reason="requires a GPU with the Pallas backend enabled (NUFFTAX_PALLAS_BACKEND)",
)


def _positions(M, seed):
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    x = jax.random.uniform(k[0], (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
    y = jax.random.uniform(k[1], (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
    return x, y


@pallas_gpu_only
def test_spread_2d_grad_xy_through_pallas_finite_diff():
    # grad w.r.t. positions of spread_2d (Pallas forward + custom VJP) must run
    # under jax.grad, stay finite, and match central finite differences.
    # Real strengths/cotangent avoid the complex-cotangent convention ambiguity.
    M, nf1, nf2 = 24, 48, 48
    x, y = _positions(M, seed=0)
    c = jax.random.normal(jax.random.PRNGKey(1), (M,)).astype(jnp.complex64)  # real values
    g = jax.random.normal(jax.random.PRNGKey(2), (nf2, nf1)).astype(jnp.float32)
    kp = compute_kernel_params(1e-6)

    @jax.jit
    def loss(x, y):
        return jnp.sum(g * jnp.real(spread_2d(x, y, c, nf1, nf2, kp)))

    gx, gy = jax.grad(loss, argnums=(0, 1))(x, y)  # runs the Pallas forward under AD
    assert np.all(np.isfinite(np.asarray(gx)))
    assert np.all(np.isfinite(np.asarray(gy)))

    eps = 1e-3
    xn, yn = np.asarray(x), np.asarray(y)
    fdx = np.zeros(M)
    fdy = np.zeros(M)
    for j in range(M):
        xp, xm = xn.copy(), xn.copy()
        xp[j] += eps
        xm[j] -= eps
        fdx[j] = (float(loss(jnp.asarray(xp), y)) - float(loss(jnp.asarray(xm), y))) / (2 * eps)
        yp, ym = yn.copy(), yn.copy()
        yp[j] += eps
        ym[j] -= eps
        fdy[j] = (float(loss(x, jnp.asarray(yp))) - float(loss(x, jnp.asarray(ym)))) / (2 * eps)

    np.testing.assert_allclose(np.asarray(gx), fdx, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(gy), fdy, rtol=2e-2, atol=2e-2)


@pallas_gpu_only
def test_nufft2d1_grad_xy_through_pallas_runs():
    # The exact case from PR #11: reverse-mode grad w.r.t. positions of nufft2d1
    # on GPU with Pallas active. Must not raise and must give finite, nonzero grads.
    M, N1, N2 = 2000, 32, 32
    x, y = _positions(M, seed=3)
    c = (jax.random.normal(jax.random.PRNGKey(4), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(5), (M,))).astype(
        jnp.complex64
    )

    def loss(x, y):
        return jnp.sum(jnp.abs(nufft2d1(x, y, c, (N1, N2), eps=1e-6)) ** 2).real

    gx, gy = jax.grad(loss, argnums=(0, 1))(x, y)

    assert np.all(np.isfinite(np.asarray(gx)))
    assert np.all(np.isfinite(np.asarray(gy)))
    assert np.linalg.norm(np.asarray(gx)) > 0
    assert np.linalg.norm(np.asarray(gy)) > 0
