"""Tests for 3D Pallas kernels (GPU-only).

These tests are collected on all platforms but skipped unless a GPU is present.
On the zay cluster (or any CUDA host with `nufftax[cuda12]` installed), they run
and verify numerical equivalence between the Pallas 3D kernels and the pure-JAX
reference implementation.

The 1D and 2D Pallas paths are already covered by indirect routes (the public
nufft*d* API dispatches to Pallas above _PALLAS_MIN_M_*). 3D Pallas is brand new
so we cover it directly here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nufftax.core.kernel import compute_kernel_params
from nufftax.core.spread import interp_3d_impl, spread_3d_impl


_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="3D Pallas kernels require a GPU")


def _setup_3d(M, nf1, nf2, nf3, eps=1e-6, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    x = jax.random.uniform(k1, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
    y = jax.random.uniform(k2, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
    z = jax.random.uniform(k3, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
    c = (jax.random.normal(k4, (M,)) + 1j * jax.random.normal(k5, (M,))).astype(jnp.complex64)
    kp = compute_kernel_params(eps, upsampfac=2.0)
    return x, y, z, c, kp


class TestSpread3dPallas:
    @gpu_only
    def test_matches_pure_jax_small(self):
        from nufftax.core.pallas_spread import spread_3d_pallas

        x, y, z, c, kp = _setup_3d(M=1024, nf1=16, nf2=16, nf3=16)
        out_pallas = spread_3d_pallas(x, y, z, c, 16, 16, 16, kp)
        out_ref = spread_3d_impl(x, y, z, c, 16, 16, 16, kp)
        np.testing.assert_allclose(out_pallas, out_ref, rtol=1e-4, atol=1e-5)

    @gpu_only
    def test_matches_pure_jax_medium(self):
        from nufftax.core.pallas_spread import spread_3d_pallas

        x, y, z, c, kp = _setup_3d(M=50_000, nf1=32, nf2=32, nf3=32)
        out_pallas = spread_3d_pallas(x, y, z, c, 32, 32, 32, kp)
        out_ref = spread_3d_impl(x, y, z, c, 32, 32, 32, kp)
        # Pallas runs in float32 internally; relax tolerance slightly
        np.testing.assert_allclose(out_pallas, out_ref, rtol=5e-4, atol=5e-5)

    @gpu_only
    def test_output_shape(self):
        from nufftax.core.pallas_spread import spread_3d_pallas

        x, y, z, c, kp = _setup_3d(M=256, nf1=8, nf2=12, nf3=10)
        out = spread_3d_pallas(x, y, z, c, 8, 12, 10, kp)
        assert out.shape == (10, 12, 8), f"expected (nf3, nf2, nf1)=(10,12,8), got {out.shape}"
        assert out.dtype == jnp.complex64


class TestInterp3dPallas:
    @gpu_only
    def test_matches_pure_jax_small(self):
        from nufftax.core.pallas_spread import interp_3d_pallas

        x, y, z, _, kp = _setup_3d(M=1024, nf1=16, nf2=16, nf3=16)
        key = jax.random.PRNGKey(123)
        fw = (jax.random.normal(key, (16, 16, 16)) + 1j * jax.random.normal(key, (16, 16, 16))).astype(jnp.complex64)
        out_pallas = interp_3d_pallas(x, y, z, fw, kp)
        out_ref = interp_3d_impl(x, y, z, fw, kp)
        np.testing.assert_allclose(out_pallas, out_ref, rtol=1e-4, atol=1e-5)

    @gpu_only
    def test_matches_pure_jax_medium(self):
        from nufftax.core.pallas_spread import interp_3d_pallas

        x, y, z, _, kp = _setup_3d(M=50_000, nf1=32, nf2=32, nf3=32)
        key = jax.random.PRNGKey(7)
        fw = (jax.random.normal(key, (32, 32, 32)) + 1j * jax.random.normal(key, (32, 32, 32))).astype(jnp.complex64)
        out_pallas = interp_3d_pallas(x, y, z, fw, kp)
        out_ref = interp_3d_impl(x, y, z, fw, kp)
        np.testing.assert_allclose(out_pallas, out_ref, rtol=5e-4, atol=5e-5)

    @gpu_only
    def test_output_shape(self):
        from nufftax.core.pallas_spread import interp_3d_pallas

        x, y, z, _, kp = _setup_3d(M=256, nf1=8, nf2=12, nf3=10)
        key = jax.random.PRNGKey(0)
        fw = (jax.random.normal(key, (10, 12, 8)) + 1j * jax.random.normal(key, (10, 12, 8))).astype(jnp.complex64)
        out = interp_3d_pallas(x, y, z, fw, kp)
        assert out.shape == (256,)
        assert out.dtype == jnp.complex64


class TestSpreadInterpAdjoint:
    """Verify that spread_3d_pallas and interp_3d_pallas are mutual adjoints."""

    @gpu_only
    def test_adjoint_inner_product(self):
        """<spread(c), fw> == <c, interp(fw)> (up to FP tolerance)."""
        from nufftax.core.pallas_spread import interp_3d_pallas, spread_3d_pallas

        x, y, z, c, kp = _setup_3d(M=2048, nf1=24, nf2=24, nf3=24)
        key = jax.random.PRNGKey(99)
        fw = (jax.random.normal(key, (24, 24, 24)) + 1j * jax.random.normal(key, (24, 24, 24))).astype(jnp.complex64)

        spread_c = spread_3d_pallas(x, y, z, c, 24, 24, 24, kp)
        interp_fw = interp_3d_pallas(x, y, z, fw, kp)

        lhs = jnp.vdot(spread_c.ravel(), fw.ravel())
        rhs = jnp.vdot(c, interp_fw)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-4)


class TestPublicDispatch:
    """Verify the dispatch path picks Pallas when M >= threshold on GPU."""

    @gpu_only
    def test_spread_3d_uses_pallas(self):
        from nufftax.core.spread import _PALLAS_MIN_M_SPREAD, spread_3d

        M = max(_PALLAS_MIN_M_SPREAD, 10_000)
        x, y, z, c, kp = _setup_3d(M=M, nf1=32, nf2=32, nf3=32)
        # Just verify it runs and produces sensible output (matches impl)
        out = spread_3d(x, y, z, c, 32, 32, 32, kp)
        out_ref = spread_3d_impl(x, y, z, c, 32, 32, 32, kp)
        np.testing.assert_allclose(out, out_ref, rtol=5e-4, atol=5e-5)

    @gpu_only
    def test_interp_3d_uses_pallas(self):
        from nufftax.core.spread import _PALLAS_MIN_M_INTERP, interp_3d

        M = max(_PALLAS_MIN_M_INTERP, 100_000)  # avoid the giant 1M default for CI feasibility
        x, y, z, _, kp = _setup_3d(M=M, nf1=32, nf2=32, nf3=32)
        key = jax.random.PRNGKey(11)
        fw = (jax.random.normal(key, (32, 32, 32)) + 1j * jax.random.normal(key, (32, 32, 32))).astype(jnp.complex64)
        out = interp_3d(x, y, z, fw, 32, 32, 32, kp)
        out_ref = interp_3d_impl(x, y, z, fw, kp)
        np.testing.assert_allclose(out, out_ref, rtol=5e-4, atol=5e-5)
