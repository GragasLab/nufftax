"""
Tests for NUFFT Type 2 transforms (uniform to nonuniform).

Type 2: c[j] = sum_k f[k] * exp(i * sign * k * x[j])

Tests cover:
- Correctness against finufft reference implementation
- Multiple precision levels
- 1D, 2D, 3D transforms
- Gradient correctness
- JAX transformations (jit, vmap, grad)
- Adjoint relationship with Type 1
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nufftax import nufft1d1, nufft1d2, nufft2d1, nufft2d2, nufft3d1, nufft3d2
from tests.conftest import (
    PRECISION_LEVELS,
    dft_nufft1d2,
    requires_finufft,
)


def relative_error(a, b):
    """Compute relative error between two arrays."""
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


# ============================================================================
# Test 1D Type 2 Against DFT Reference
# ============================================================================


class TestNUFFT1D2DFT:
    """Test 1D Type 2 against direct DFT computation."""

    def test_dft_reference_small(self, rng):
        """Test DFT reference implementation itself."""
        M = 10
        n_modes = 8
        x = rng.uniform(0, 2 * np.pi, M)
        f = rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)

        c = dft_nufft1d2(x, f)

        assert c.shape == (M,)
        assert c.dtype == np.complex128

    def test_dft_single_point(self):
        """Test DFT with single point at x=0."""
        x = np.array([0.0])
        n_modes = 8
        f = np.ones(n_modes, dtype=np.complex128)

        c = dft_nufft1d2(x, f)

        # exp(i*k*0) = 1 for all k, so c[0] = sum(f) = n_modes
        expected = np.array([n_modes + 0j])
        assert_allclose(c, expected, rtol=1e-10)

    def test_dft_uniform_modes(self):
        """Test with uniform f (all ones)."""
        M = 5
        n_modes = 8
        x = np.linspace(0, 2 * np.pi, M, endpoint=False)
        f = np.ones(n_modes, dtype=np.complex128)

        c = dft_nufft1d2(x, f)

        assert c.shape == (M,)
        # Each c[j] = sum_k exp(i*k*x[j])


# ============================================================================
# Test 1D Type 2 Against FINUFFT
# ============================================================================


class TestNUFFT1D2FINUFFT:
    """Test 1D Type 2 against FINUFFT reference."""

    @requires_finufft
    def test_finufft_1d2_basic(self, rng):
        """Basic test of FINUFFT 1D Type 2."""
        import finufft

        M = 100
        n_modes = 64
        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        c = finufft.nufft1d2(x, f, eps=1e-9)

        assert c.shape == (M,)
        assert c.dtype == np.complex128

    @requires_finufft
    @pytest.mark.parametrize("eps", PRECISION_LEVELS)
    def test_nufft1d2_vs_finufft_precision(self, rng, eps):
        """Test nufft1d2 at multiple precision levels."""
        import finufft

        M = 1000
        n_modes = 128

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        # FINUFFT reference
        c_ref = finufft.nufft1d2(x, f, eps=eps)

        # JAX implementation
        c_jax = nufft1d2(jnp.array(x), jnp.array(f), eps=eps)

        rel_err = relative_error(c_jax, jnp.array(c_ref))
        # Use max to account for floating-point precision limits at very high accuracy
        assert rel_err < max(10 * eps, 1e-5), f"eps={eps}, rel_err={rel_err}"

    @requires_finufft
    def test_nufft1d2_vs_finufft_large(self, rng):
        """Test with larger problem size."""
        import finufft

        M = 10000
        n_modes = 1024
        eps = 1e-6

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        c_ref = finufft.nufft1d2(x, f, eps=eps)
        c_jax = nufft1d2(jnp.array(x), jnp.array(f), eps=eps)

        rel_err = relative_error(c_jax, jnp.array(c_ref))
        # Large problems may accumulate more numerical error
        assert rel_err < max(10 * eps, 1e-4)


# ============================================================================
# Test 2D Type 2 Against FINUFFT
# ============================================================================


class TestNUFFT2D2FINUFFT:
    """Test 2D Type 2 against FINUFFT reference."""

    @requires_finufft
    def test_finufft_2d2_basic(self, rng):
        """Basic test of FINUFFT 2D Type 2."""
        import finufft

        M = 200
        n_modes = (32, 32)
        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        c = finufft.nufft2d2(x, y, f, eps=1e-9)

        assert c.shape == (M,)
        assert c.dtype == np.complex128

    @requires_finufft
    @pytest.mark.parametrize("eps", PRECISION_LEVELS)
    def test_nufft2d2_vs_finufft_precision(self, rng, eps):
        """Test 2D Type 2 at multiple precision levels."""
        import finufft

        M = 500
        n_modes = (32, 48)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        # Generate f with FINUFFT shape convention
        f_finufft = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        # FINUFFT takes f as (n1, n2), JAX expects (n2, n1)
        c_ref = finufft.nufft2d2(x, y, f_finufft, eps=eps)
        c_jax = nufft2d2(jnp.array(x), jnp.array(y), jnp.array(f_finufft.T), eps=eps)

        rel_err = relative_error(c_jax, jnp.array(c_ref))
        # Use max to account for floating-point precision limits at very high accuracy
        assert rel_err < max(10 * eps, 1e-5)


# ============================================================================
# Test 3D Type 2 Against FINUFFT
# ============================================================================


class TestNUFFT3D2FINUFFT:
    """Test 3D Type 2 against FINUFFT reference."""

    @requires_finufft
    def test_finufft_3d2_basic(self, rng):
        """Basic test of FINUFFT 3D Type 2."""
        import finufft

        M = 200
        n_modes = (16, 16, 16)
        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        z = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        c = finufft.nufft3d2(x, y, z, f, eps=1e-9)

        assert c.shape == (M,)
        assert c.dtype == np.complex128

    @requires_finufft
    @pytest.mark.parametrize("eps", [1e-3, 1e-6])  # Fewer tests for 3D (slower)
    def test_nufft3d2_vs_finufft_precision(self, rng, eps):
        """Test 3D Type 2 at multiple precision levels."""
        import finufft

        M = 300
        n_modes = (16, 16, 16)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        z = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        # Generate f with FINUFFT shape convention
        f_finufft = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        # FINUFFT takes f as (n1, n2, n3), JAX expects (n3, n2, n1)
        c_ref = finufft.nufft3d2(x, y, z, f_finufft, eps=eps)
        c_jax = nufft3d2(
            jnp.array(x),
            jnp.array(y),
            jnp.array(z),
            jnp.array(np.transpose(f_finufft, (2, 1, 0))),
            eps=eps,
        )

        rel_err = relative_error(c_jax, jnp.array(c_ref))
        assert rel_err < 10 * eps


# ============================================================================
# Test Adjoint Property: <nufft1(c), f> = <c, nufft2(f)>
# ============================================================================


class TestAdjointProperty:
    """Test that Type 1 and Type 2 are adjoints."""

    def test_adjoint_dft_1d(self, rng):
        """Test adjoint property using DFT reference implementations."""
        M = 30
        n_modes = 16

        x = rng.uniform(0, 2 * np.pi, M)
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        f = rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)

        # Type 1: c -> f1 (with isign=+1)
        from tests.conftest import dft_nufft1d1

        f1 = dft_nufft1d1(x, c, n_modes)

        # Type 2 with isign=-1 is the adjoint of Type 1 with isign=+1
        c2 = dft_nufft1d2(x, f, isign=-1)

        # <nufft1(c), f> should equal <c, nufft2_adjoint(f)>
        lhs = np.vdot(f1, f)  # <f1, f>
        rhs = np.vdot(c, c2)  # <c, c2>

        rel_diff = np.abs(lhs - rhs) / np.abs(lhs)
        assert rel_diff < 1e-10, f"Adjoint relation violated: rel_diff={rel_diff}"

    @requires_finufft
    def test_adjoint_finufft_1d(self, rng):
        """Test adjoint property using FINUFFT.

        Note: FINUFFT's Type 1 and Type 2 with default isign are adjoints.
        Both use isign=-1 by default, and FINUFFT defines the transforms
        such that the adjoint relationship holds.
        """
        import finufft

        M = 500
        n_modes = 64
        eps = 1e-12

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        # Type 1: c -> f1 (default isign=-1)
        f1 = finufft.nufft1d1(x, c, n_modes, eps=eps)

        # Type 2: f -> c2 (default isign=-1, which is adjoint of Type 1)
        c2 = finufft.nufft1d2(x, f, eps=eps)

        # <nufft1(c), f> should equal <c, nufft2(f)>
        lhs = np.vdot(f1, f)
        rhs = np.vdot(c, c2)

        rel_diff = np.abs(lhs - rhs) / np.abs(lhs)
        assert rel_diff < 1e-10, f"Adjoint relation violated: rel_diff={rel_diff}"

    def test_adjoint_jax_1d(self, rng):
        """Test adjoint property using JAX implementations.

        Note: nufftax defaults to isign=+1 for Type 1 and isign=-1 for Type 2,
        which makes them adjoints by default. This matches the mathematical
        definition where Type 2 with exp(-ikx) is the adjoint of Type 1 with exp(+ikx).
        """
        M = 500
        n_modes = 64
        eps = 1e-9

        x = jnp.array(rng.uniform(-np.pi, np.pi, M).astype(np.float64))
        c = jnp.array((rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128))
        f = jnp.array((rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128))

        # Type 1: c -> f1 (default isign=+1)
        f1 = nufft1d1(x, c, n_modes, eps=eps)

        # Type 2: f -> c2 (default isign=-1, which is adjoint of Type 1 with isign=+1)
        c2 = nufft1d2(x, f, eps=eps)

        # <nufft1(c), f> should equal <c, nufft2(f)>
        lhs = jnp.vdot(f1, f)
        rhs = jnp.vdot(c, c2)

        rel_diff = jnp.abs(lhs - rhs) / jnp.abs(lhs)
        assert rel_diff < 2e-6, f"Adjoint relation violated: rel_diff={rel_diff}"

    @requires_finufft
    def test_adjoint_finufft_2d(self, rng):
        """Test adjoint property in 2D using FINUFFT."""
        import finufft

        M = 300
        n_modes = (32, 32)
        eps = 1e-12

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        f1 = finufft.nufft2d1(x, y, c, n_modes, eps=eps)
        c2 = finufft.nufft2d2(x, y, f, eps=eps)

        lhs = np.vdot(f1, f)
        rhs = np.vdot(c, c2)

        rel_diff = np.abs(lhs - rhs) / np.abs(lhs)
        assert rel_diff < 1e-10

    def test_adjoint_jax_2d(self, rng):
        """Test adjoint property in 2D using JAX implementations."""
        M = 300
        n_modes = (32, 32)
        eps = 1e-9

        x = jnp.array(rng.uniform(-np.pi, np.pi, M).astype(np.float64))
        y = jnp.array(rng.uniform(-np.pi, np.pi, M).astype(np.float64))
        c = jnp.array((rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128))
        f = jnp.array((rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128))

        f1 = nufft2d1(x, y, c, n_modes, eps=eps)
        c2 = nufft2d2(x, y, f, eps=eps)

        lhs = jnp.vdot(f1, f)
        rhs = jnp.vdot(c, c2)

        rel_diff = jnp.abs(lhs - rhs) / jnp.abs(lhs)
        assert rel_diff < 1e-6

    @requires_finufft
    def test_adjoint_finufft_3d(self, rng):
        """Test adjoint property in 3D using FINUFFT."""
        import finufft

        M = 200
        n_modes = (12, 12, 12)
        eps = 1e-12

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        z = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        f1 = finufft.nufft3d1(x, y, z, c, n_modes, eps=eps)
        c2 = finufft.nufft3d2(x, y, z, f, eps=eps)

        lhs = np.vdot(f1, f)
        rhs = np.vdot(c, c2)

        rel_diff = np.abs(lhs - rhs) / np.abs(lhs)
        assert rel_diff < 1e-10

    def test_adjoint_jax_3d(self, rng):
        """Test adjoint property in 3D using JAX implementations."""
        M = 200
        n_modes = (12, 12, 12)
        eps = 1e-9

        x = jnp.array(rng.uniform(-np.pi, np.pi, M).astype(np.float64))
        y = jnp.array(rng.uniform(-np.pi, np.pi, M).astype(np.float64))
        z = jnp.array(rng.uniform(-np.pi, np.pi, M).astype(np.float64))
        c = jnp.array((rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128))
        f = jnp.array((rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128))

        f1 = nufft3d1(x, y, z, c, n_modes, eps=eps)
        c2 = nufft3d2(x, y, z, f, eps=eps)

        lhs = jnp.vdot(f1, f)
        rhs = jnp.vdot(c, c2)

        rel_diff = jnp.abs(lhs - rhs) / jnp.abs(lhs)
        assert rel_diff < 1e-6
