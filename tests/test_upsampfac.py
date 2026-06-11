"""Tests for the ``upsampfac`` option on Type 1 / Type 2 transforms.

Covers: scalar default, scalar 1.25 (works but less accurate with the generic
kernel), and the ``(forward, backward)`` tuple form that sets the oversampling of
the forward transform and its adjoint (used in reverse-mode AD) independently.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nufftax import nufft1d1, nufft1d2, nufft2d1


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _dft_1d1(x, c, n):
    k = np.arange(-(n // 2), (n + 1) // 2)
    return np.exp(1j * np.outer(k, np.asarray(x))) @ np.asarray(c)


def test_default_equals_explicit_2(rng):
    M, N = 200, 64
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = (jnp.asarray(rng.standard_normal(M)) + 1j * jnp.asarray(rng.standard_normal(M))).astype(jnp.complex64)
    assert_allclose(np.asarray(nufft1d1(x, c, N)), np.asarray(nufft1d1(x, c, N, upsampfac=2.0)), rtol=1e-4, atol=1e-6)
    y = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    assert_allclose(
        np.asarray(nufft2d1(x, y, c, (N, N))),
        np.asarray(nufft2d1(x, y, c, (N, N), upsampfac=2.0)),
        rtol=1e-4,
        atol=1e-6,
    )


def test_forward_accuracy_vs_upsampfac(rng):
    M, N = 300, 64
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = (jnp.asarray(rng.standard_normal(M)) + 1j * jnp.asarray(rng.standard_normal(M))).astype(jnp.complex64)
    ref = _dft_1d1(x, c, N)

    def relerr(uf):
        f = np.asarray(nufft1d1(x, c, N, eps=1e-6, upsampfac=uf))
        return np.max(np.abs(f - ref)) / np.max(np.abs(ref))

    err2, err125 = relerr(2.0), relerr(1.25)
    assert err2 < 1e-4
    assert err125 < 1e-2  # the knob works; the generic kernel is just less accurate at 1.25
    assert err125 > err2


def test_tuple_forward_uses_first_element(rng):
    M, N = 200, 64
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = (jnp.asarray(rng.standard_normal(M)) + 1j * jnp.asarray(rng.standard_normal(M))).astype(jnp.complex64)
    a = nufft1d1(x, c, N, upsampfac=(2.0, 1.25))  # forward uses 2.0
    b = nufft1d1(x, c, N, upsampfac=2.0)
    assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6)


def test_grad_default_matches_explicit_and_tuple_runs(rng):
    M, N = 100, 64
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    c = (jnp.asarray(rng.standard_normal(M)) + 1j * jnp.asarray(rng.standard_normal(M))).astype(jnp.complex64)

    def loss(c, uf):
        return jnp.sum(jnp.abs(nufft1d1(x, c, N, upsampfac=uf)) ** 2).real

    g_scalar = jax.grad(loss)(c, 2.0)
    g_default = jax.grad(lambda c: jnp.sum(jnp.abs(nufft1d1(x, c, N)) ** 2).real)(c)
    assert_allclose(np.asarray(g_scalar), np.asarray(g_default), rtol=1e-5, atol=1e-5)

    g_tuple = jax.grad(loss)(c, (2.0, 1.25))  # adjoint at a different upsampfac
    assert np.all(np.isfinite(np.asarray(g_tuple)))


def test_type2_upsampfac_forward_and_grad(rng):
    M, N = 200, 64
    x = jnp.asarray(rng.uniform(-np.pi, np.pi, M))
    f = (jnp.asarray(rng.standard_normal(N)) + 1j * jnp.asarray(rng.standard_normal(N))).astype(jnp.complex64)
    assert_allclose(np.asarray(nufft1d2(x, f, upsampfac=2.0)), np.asarray(nufft1d2(x, f)), rtol=1e-4, atol=1e-6)
    g = jax.grad(lambda f: jnp.sum(jnp.abs(nufft1d2(x, f, upsampfac=(2.0, 1.25))) ** 2).real)(f)
    assert np.all(np.isfinite(np.asarray(g)))
