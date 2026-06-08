"""
Tests for user-supplied custom spreading/interpolation kernels.

Covers the ``Kernel`` abstraction that lets ``spread_*`` / ``interp_*`` use an
arbitrary kernel (e.g. a Gaussian) instead of the built-in ES kernel:
- the ES kernel routed through ``Kernel`` matches the ``KernelParams`` path,
- a Gaussian kernel matches an independent direct reference,
- gradients w.r.t. coordinates match finite differences,
- spreading and interpolation are adjoints for the same kernel,
- the autodiff derivative fallback matches an analytic ``phi_and_dphi``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nufftax.core import (
    interp_1d,
    spread_1d,
    spread_2d,
    spread_3d,
)
from nufftax.core.kernel import Kernel, compute_kernel_params, es_kernel_spec
from nufftax.core.spread import spread_1d_impl, spread_2d_impl, spread_3d_impl


NSPREAD = 10
SIGMA = 1.5

_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="custom-kernel Pallas path requires a GPU")


def gaussian_kernel(nspread, sigma):
    """Truncated Gaussian Kernel ``phi(z) = exp(-z^2 / (2*sigma^2))`` for tests."""
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    inv_s2 = 1.0 / (sigma * sigma)

    def phi(z):
        return jnp.exp(-inv_2s2 * (z * z))

    def phi_and_dphi(z):
        p = phi(z)
        return p, -inv_s2 * z * p

    return Kernel(nspread=nspread, phi=phi, phi_and_dphi=phi_and_dphi)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def kernel_params():
    return compute_kernel_params(1e-6)


# ============================================================================
# Independent direct references (pure numpy)
# ============================================================================


def _fold_rescale(x, n):
    r = x / (2.0 * np.pi) + 0.5
    return (r - np.floor(r)) * n


def _gauss(z, sigma):
    return np.exp(-(z * z) / (2.0 * sigma * sigma))


def spread_1d_gauss_direct(x, c, nf, nspread, sigma):
    fw = np.zeros(nf, dtype=np.complex128)
    xs = _fold_rescale(x, nf)
    i0 = np.ceil(xs - nspread / 2.0).astype(np.int64)
    for j in range(len(x)):
        for k in range(nspread):
            idx = (i0[j] + k) % nf
            z = (i0[j] + k) - xs[j]
            fw[idx] += c[j] * _gauss(z, sigma)
    return fw


def spread_2d_gauss_direct(x, y, c, nf1, nf2, nspread, sigma):
    fw = np.zeros((nf2, nf1), dtype=np.complex128)
    xs = _fold_rescale(x, nf1)
    ys = _fold_rescale(y, nf2)
    i0x = np.ceil(xs - nspread / 2.0).astype(np.int64)
    i0y = np.ceil(ys - nspread / 2.0).astype(np.int64)
    for j in range(len(x)):
        for ky in range(nspread):
            iy = (i0y[j] + ky) % nf2
            wy = _gauss((i0y[j] + ky) - ys[j], sigma)
            for kx in range(nspread):
                ix = (i0x[j] + kx) % nf1
                wx = _gauss((i0x[j] + kx) - xs[j], sigma)
                fw[iy, ix] += c[j] * wy * wx
    return fw


# ============================================================================
# ES kernel routed through Kernel matches the KernelParams path
# ============================================================================


def test_es_via_kernel_matches_kernelparams_1d(rng, kernel_params):
    M, nf = 200, 128
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M))

    ref = spread_1d(x, c, nf, kernel_params)
    via = spread_1d(x, c, nf, es_kernel_spec(kernel_params))
    assert_allclose(np.asarray(via), np.asarray(ref), rtol=1e-6, atol=1e-6)


def test_es_via_kernel_matches_kernelparams_2d(rng, kernel_params):
    M, nf1, nf2 = 200, 64, 48
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    y = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M))

    ref = spread_2d(x, y, c, nf1, nf2, kernel_params)
    via = spread_2d(x, y, c, nf1, nf2, es_kernel_spec(kernel_params))
    assert_allclose(np.asarray(via), np.asarray(ref), rtol=1e-6, atol=1e-6)


# ============================================================================
# Gaussian custom kernel matches direct reference
# ============================================================================


def test_gaussian_spread_1d_matches_direct(rng):
    M, nf = 150, 100
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M))
    kernel = gaussian_kernel(NSPREAD, SIGMA)

    out = np.asarray(spread_1d(x, c, nf, kernel))
    ref = spread_1d_gauss_direct(np.asarray(x), np.asarray(c), nf, NSPREAD, SIGMA)
    assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_gaussian_spread_2d_matches_direct(rng):
    M, nf1, nf2 = 120, 40, 32
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    y = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M))
    kernel = gaussian_kernel(NSPREAD, SIGMA)

    out = np.asarray(spread_2d(x, y, c, nf1, nf2, kernel))
    ref = spread_2d_gauss_direct(np.asarray(x), np.asarray(y), np.asarray(c), nf1, nf2, NSPREAD, SIGMA)
    assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_gaussian_spread_3d_runs_and_is_finite(rng):
    M, nf1, nf2, nf3 = 100, 24, 20, 16
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    y = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    z = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M))
    kernel = gaussian_kernel(NSPREAD, SIGMA)

    out = spread_3d(x, y, z, c, nf1, nf2, nf3, kernel)
    assert out.shape == (nf3, nf2, nf1)
    assert np.all(np.isfinite(np.asarray(out).view(np.float64)))


# ============================================================================
# Spread / interp are adjoints for the same custom kernel
# ============================================================================


def test_custom_kernel_adjoint_1d(rng):
    M, nf = 80, 64
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M))
    g = jnp.asarray(rng.standard_normal(nf) + 1j * rng.standard_normal(nf))
    kernel = gaussian_kernel(NSPREAD, SIGMA)

    # <g, spread(c)> == <interp(g), c>
    lhs = np.vdot(np.asarray(g), np.asarray(spread_1d(x, c, nf, kernel)))
    rhs = np.vdot(np.asarray(interp_1d(x, g, nf, kernel)), np.asarray(c))
    assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-5)


# ============================================================================
# Gradient w.r.t. coordinates matches finite differences
# ============================================================================


def test_custom_kernel_grad_x_1d_finite_difference(rng):
    # Real inputs avoid the complex-cotangent convention ambiguity so the
    # gradient can be compared directly against central finite differences.
    M, nf = 30, 64
    x = jnp.asarray(rng.uniform(-2.0, 2.0, M))
    c = jnp.asarray(rng.standard_normal(M)).astype(jnp.complex64)
    g = jnp.asarray(rng.standard_normal(nf))
    kernel = gaussian_kernel(NSPREAD, SIGMA)

    def loss(xv):
        return jnp.sum(g * jnp.real(spread_1d(xv, c, nf, kernel)))

    grad = np.asarray(jax.grad(loss)(x))

    eps = 1e-4
    fd = np.zeros(M)
    x_np = np.asarray(x)
    for j in range(M):
        xp = x_np.copy()
        xp[j] += eps
        xm = x_np.copy()
        xm[j] -= eps
        fd[j] = (float(loss(jnp.asarray(xp))) - float(loss(jnp.asarray(xm)))) / (2 * eps)

    assert_allclose(grad, fd, rtol=2e-2, atol=1e-1)


# ============================================================================
# Autodiff derivative fallback matches an analytic phi_and_dphi
# ============================================================================


def test_autodiff_derivative_fallback_matches_analytic(rng):
    M, nf = 30, 64
    x = jnp.asarray(rng.uniform(-2.0, 2.0, M))
    c = jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M))
    g = jnp.asarray(rng.standard_normal(nf) + 1j * rng.standard_normal(nf))

    inv_2s2 = 1.0 / (2.0 * SIGMA * SIGMA)

    def phi(zz):
        return jnp.exp(-inv_2s2 * (zz * zz))

    analytic = gaussian_kernel(NSPREAD, SIGMA)  # has phi_and_dphi
    autodiff = Kernel(nspread=NSPREAD, phi=phi)  # derivative via autodiff

    def loss(kernel, xv):
        return jnp.real(jnp.vdot(g, spread_1d(xv, c, nf, kernel)))

    grad_analytic = np.asarray(jax.grad(lambda xv: loss(analytic, xv))(x))
    grad_autodiff = np.asarray(jax.grad(lambda xv: loss(autodiff, xv))(x))
    assert_allclose(grad_autodiff, grad_analytic, rtol=1e-5, atol=1e-6)


# ============================================================================
# NUFFTAX_PALLAS_BACKEND env var gates the Pallas spreading backend (default on)
# ============================================================================


def test_pallas_backend_env_flag(monkeypatch):
    from nufftax.core.spread import _bool_env

    monkeypatch.delenv("NUFFTAX_PALLAS_BACKEND", raising=False)
    assert _bool_env("NUFFTAX_PALLAS_BACKEND", True) is True  # default on
    for off in ("0", "false", "False", "no", "off"):
        monkeypatch.setenv("NUFFTAX_PALLAS_BACKEND", off)
        assert _bool_env("NUFFTAX_PALLAS_BACKEND", True) is False
    for on in ("1", "true", "yes", "on"):
        monkeypatch.setenv("NUFFTAX_PALLAS_BACKEND", on)
        assert _bool_env("NUFFTAX_PALLAS_BACKEND", True) is True


# ============================================================================
# GPU-only: the custom kernel runs through the Pallas spread kernels and
# matches the pure-JAX reference (the point that a custom phi can be threaded
# into Pallas, not just the built-in ES kernel).
# ============================================================================


def _setup(M, ndim, seed=0):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, ndim + 2)
    coords = [
        jax.random.uniform(keys[i], (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32) for i in range(ndim)
    ]
    c = (jax.random.normal(keys[-2], (M,)) + 1j * jax.random.normal(keys[-1], (M,))).astype(jnp.complex64)
    return coords, c


class TestCustomKernelPallas:
    @gpu_only
    def test_pallas_1d_matches_pure_jax(self):
        from nufftax.core.pallas_spread import spread_1d_pallas

        kernel = gaussian_kernel(NSPREAD, SIGMA)
        (x,), c = _setup(20_000, 1)
        out_pallas = spread_1d_pallas(x, c, 256, kernel)
        out_ref = spread_1d_impl(x, c, 256, kernel)
        np.testing.assert_allclose(np.asarray(out_pallas), np.asarray(out_ref), rtol=5e-4, atol=5e-5)

    @gpu_only
    def test_pallas_2d_matches_pure_jax(self):
        from nufftax.core.pallas_spread import spread_2d_pallas

        kernel = gaussian_kernel(NSPREAD, SIGMA)
        (x, y), c = _setup(20_000, 2)
        out_pallas = spread_2d_pallas(x, y, c, 64, 48, kernel)
        out_ref = spread_2d_impl(x, y, c, 64, 48, kernel)
        np.testing.assert_allclose(np.asarray(out_pallas), np.asarray(out_ref), rtol=5e-4, atol=5e-5)

    @gpu_only
    def test_pallas_3d_matches_pure_jax(self):
        from nufftax.core.pallas_spread import spread_3d_pallas

        kernel = gaussian_kernel(NSPREAD, SIGMA)
        (x, y, z), c = _setup(20_000, 3)
        out_pallas = spread_3d_pallas(x, y, z, c, 24, 20, 16, kernel)
        out_ref = spread_3d_impl(x, y, z, c, 24, 20, 16, kernel)
        np.testing.assert_allclose(np.asarray(out_pallas), np.asarray(out_ref), rtol=5e-4, atol=5e-5)

    @gpu_only
    def test_public_dispatch_routes_custom_kernel_to_pallas(self):
        # With the Pallas backend enabled (default), spread_1d with a custom
        # kernel must take the Pallas path and still match the pure-JAX reference.
        kernel = gaussian_kernel(NSPREAD, SIGMA)
        (x,), c = _setup(20_000, 1)
        out = spread_1d(x, c, 256, kernel)
        out_ref = spread_1d_impl(x, c, 256, kernel)
        np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), rtol=5e-4, atol=5e-5)
