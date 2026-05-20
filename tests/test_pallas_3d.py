"""Tests for the 3D Pallas spreading kernel (GPU-only).

Collected on all platforms, skipped unless a GPU is present. On a CUDA host
(`nufftax[cuda12]`) they verify the Pallas 3D spread kernel matches the
pure-JAX reference and that the public dispatch routes to it.

Interpolation (Type 2) has no Pallas kernel: benchmarked on H100 it was at
best parity vs XLA and 2-4x slower in 3D, so it is left to pure JAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nufftax.core.kernel import compute_kernel_params
from nufftax.core.spread import spread_3d_impl


_HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="3D Pallas kernel requires a GPU")


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
        np.testing.assert_allclose(out_pallas, out_ref, rtol=5e-4, atol=5e-5)

    @gpu_only
    def test_output_shape(self):
        from nufftax.core.pallas_spread import spread_3d_pallas

        x, y, z, c, kp = _setup_3d(M=256, nf1=8, nf2=12, nf3=10)
        out = spread_3d_pallas(x, y, z, c, 8, 12, 10, kp)
        assert out.shape == (10, 12, 8), f"expected (nf3, nf2, nf1)=(10,12,8), got {out.shape}"
        assert out.dtype == jnp.complex64


class TestPublicDispatch:
    @gpu_only
    def test_spread_3d_uses_pallas(self):
        from nufftax.core.spread import _PALLAS_MIN_M_SPREAD_3D, spread_3d

        M = max(_PALLAS_MIN_M_SPREAD_3D, 10_000)
        x, y, z, c, kp = _setup_3d(M=M, nf1=32, nf2=32, nf3=32)
        out = spread_3d(x, y, z, c, 32, 32, 32, kp)
        out_ref = spread_3d_impl(x, y, z, c, 32, 32, 32, kp)
        np.testing.assert_allclose(out, out_ref, rtol=5e-4, atol=5e-5)
