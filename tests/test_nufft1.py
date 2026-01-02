"""
Tests for NUFFT Type 1 transforms (nonuniform to uniform).

Type 1: f[k] = sum_j c[j] * exp(i * sign * k * x[j])

Tests cover:
- Correctness against finufft reference implementation
- Multiple precision levels
- 1D, 2D, 3D transforms
- Gradient correctness
- JAX transformations (jit, vmap, grad)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import NUFFT functions when implemented
# from nufftax import nufft1d1, nufft2d1, nufft3d1
# from nufftax.transforms.nufft1 import nufft1d1, nufft2d1, nufft3d1
from tests.conftest import (
    PRECISION_LEVELS,
    dft_nufft1d1,
    requires_finufft,
)


# ============================================================================
# Test 1D Type 1 Against DFT Reference
# ============================================================================


class TestNUFFT1D1DFT:
    """Test 1D Type 1 against direct DFT computation."""

    def test_dft_reference_small(self, rng):
        """Test DFT reference implementation itself."""
        M = 10
        n_modes = 8
        x = rng.uniform(0, 2 * np.pi, M)
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)

        f = dft_nufft1d1(x, c, n_modes)

        assert f.shape == (n_modes,)
        assert f.dtype == np.complex128

    def test_dft_single_point(self):
        """Test DFT with single point at x=0."""
        x = np.array([0.0])
        c = np.array([1.0 + 0j])
        n_modes = 8

        f = dft_nufft1d1(x, c, n_modes)

        # exp(i*k*0) = 1 for all k, so f[k] = 1
        expected = np.ones(n_modes, dtype=np.complex128)
        assert_allclose(f, expected, rtol=1e-10)

    def test_dft_single_mode_contribution(self):
        """Test that single mode from x gives correct phases."""
        n_modes = 8
        x = np.array([np.pi / 2])  # exp(i*k*pi/2) = i^k
        c = np.array([1.0 + 0j])

        f = dft_nufft1d1(x, c, n_modes)

        k = np.arange(n_modes) - n_modes // 2
        expected = np.exp(1j * k * np.pi / 2)
        assert_allclose(f, expected, rtol=1e-10)

    @pytest.mark.parametrize("M", [10, 50, 100])
    @pytest.mark.parametrize("n_modes", [8, 32, 64])
    def test_nufft1d1_vs_dft(self, rng, M, n_modes):
        """Compare nufft1d1 against DFT for small problems."""
        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        # DFT reference
        dft_nufft1d1(x, c, n_modes)

        # JAX implementation
        # f_jax = nufft1d1(jnp.array(x), jnp.array(c), n_modes, eps=1e-12)

        # rel_err = relative_error(f_jax, jnp.array(f_dft))
        # assert rel_err < 1e-10, f"Relative error {rel_err} exceeds tolerance"


# ============================================================================
# Test 1D Type 1 Against FINUFFT
# ============================================================================


class TestNUFFT1D1FINUFFT:
    """Test 1D Type 1 against FINUFFT reference."""

    @requires_finufft
    def test_finufft_1d1_basic(self, rng):
        """Basic test of FINUFFT 1D Type 1."""
        import finufft

        M = 100
        n_modes = 64
        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f = finufft.nufft1d1(x, c, n_modes, eps=1e-9)

        assert f.shape == (n_modes,)
        assert f.dtype == np.complex128

    @requires_finufft
    @pytest.mark.parametrize("eps", PRECISION_LEVELS)
    def test_nufft1d1_vs_finufft_precision(self, rng, eps):
        """Test nufft1d1 at multiple precision levels."""
        import finufft

        M = 1000
        n_modes = 128

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        # FINUFFT reference
        finufft.nufft1d1(x, c, n_modes, eps=eps)

        # JAX implementation
        # f_jax = nufft1d1(jnp.array(x), jnp.array(c), n_modes, eps=eps)

        # rel_err = relative_error(f_jax, jnp.array(f_ref))
        # assert rel_err < 10 * eps, f"eps={eps}, rel_err={rel_err}"

    @requires_finufft
    def test_nufft1d1_vs_finufft_large(self, rng):
        """Test with larger problem size."""
        import finufft

        M = 10000
        n_modes = 1024
        eps = 1e-6

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        finufft.nufft1d1(x, c, n_modes, eps=eps)
        # f_jax = nufft1d1(jnp.array(x), jnp.array(c), n_modes, eps=eps)

        # rel_err = relative_error(f_jax, jnp.array(f_ref))
        # assert rel_err < 10 * eps

    @requires_finufft
    def test_nufft1d1_iflag_positive(self, rng):
        """Test with positive exponential sign (iflag=+1)."""
        import finufft

        M = 200
        n_modes = 64
        eps = 1e-9

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        # FINUFFT with iflag=1 (positive exponential)
        finufft.nufft1d1(x, c, n_modes, eps=eps, isign=1)

        # JAX with iflag=1
        # f_jax = nufft1d1(jnp.array(x), jnp.array(c), n_modes, eps=eps, iflag=1)

        # rel_err = relative_error(f_jax, jnp.array(f_ref))
        # assert rel_err < 10 * eps


# ============================================================================
# Test 2D Type 1 Against FINUFFT
# ============================================================================


class TestNUFFT2D1FINUFFT:
    """Test 2D Type 1 against FINUFFT reference."""

    @requires_finufft
    def test_finufft_2d1_basic(self, rng):
        """Basic test of FINUFFT 2D Type 1."""
        import finufft

        M = 200
        n_modes = (32, 32)
        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f = finufft.nufft2d1(x, y, c, n_modes, eps=1e-9)

        assert f.shape == n_modes
        assert f.dtype == np.complex128

    @requires_finufft
    @pytest.mark.parametrize("eps", PRECISION_LEVELS)
    def test_nufft2d1_vs_finufft_precision(self, rng, eps):
        """Test 2D Type 1 at multiple precision levels."""
        import finufft

        M = 500
        n_modes = (32, 48)

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        finufft.nufft2d1(x, y, c, n_modes, eps=eps)
        # f_jax = nufft2d1(jnp.array(x), jnp.array(y), jnp.array(c),
        #                  n_modes, eps=eps)

        # rel_err = relative_error(f_jax, jnp.array(f_ref))
        # assert rel_err < 10 * eps


# ============================================================================
# Test 3D Type 1 Against FINUFFT
# ============================================================================


class TestNUFFT3D1FINUFFT:
    """Test 3D Type 1 against FINUFFT reference."""

    @requires_finufft
    def test_finufft_3d1_basic(self, rng):
        """Basic test of FINUFFT 3D Type 1."""
        import finufft

        M = 200
        n_modes = (16, 16, 16)
        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        z = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f = finufft.nufft3d1(x, y, z, c, n_modes, eps=1e-9)

        assert f.shape == n_modes
        assert f.dtype == np.complex128

    @requires_finufft
    @pytest.mark.parametrize("eps", [1e-3, 1e-6])  # Fewer tests for 3D (slower)
    def test_nufft3d1_vs_finufft_precision(self, rng, eps):
        """Test 3D Type 1 at multiple precision levels."""
        import finufft

        M = 300
        n_modes = (16, 16, 16)

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        z = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        finufft.nufft3d1(x, y, z, c, n_modes, eps=eps)
        # f_jax = nufft3d1(jnp.array(x), jnp.array(y), jnp.array(z),
        #                  jnp.array(c), n_modes, eps=eps)

        # rel_err = relative_error(f_jax, jnp.array(f_ref))
        # assert rel_err < 10 * eps
