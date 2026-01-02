"""
Tests for edge cases and special inputs.
Based on FINUFFT test coverage.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nufftax.transforms import nufft1d1, nufft2d1, nufft3d1


class TestEmptyInputs:
    """Test behavior with empty or minimal inputs."""

    def test_single_point(self):
        """Test NUFFT with a single nonuniform point."""
        x = jnp.array([0.5])
        c = jnp.array([1.0 + 1j])
        N = 16

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.shape == (N,)
        # Single point: f[k] = c[0] * exp(i*k*x[0])
        k = jnp.arange(-N // 2, (N + 1) // 2)
        expected = c[0] * jnp.exp(1j * k * x[0])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_two_points(self):
        """Test NUFFT with two nonuniform points."""
        x = jnp.array([0.0, jnp.pi])
        c = jnp.array([1.0 + 0j, 1.0 + 0j])
        N = 8

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.shape == (N,)
        # Two symmetric points should have real output for even k
        # f[k] = exp(i*k*0) + exp(i*k*pi) = 1 + (-1)^k
        k = jnp.arange(-N // 2, (N + 1) // 2)
        expected = 1 + jnp.cos(k * jnp.pi)  # exp(i*k*pi) = cos(k*pi)
        np.testing.assert_allclose(result.real, expected, rtol=1e-5)
        np.testing.assert_allclose(result.imag, jnp.zeros_like(result.imag), atol=1e-6)


class TestSpecialCoordinates:
    """Test with special coordinate values."""

    def test_points_at_origin(self):
        """Test with all points at x=0."""
        M, N = 10, 32
        x = jnp.zeros(M)
        c = jnp.ones(M, dtype=jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        # All points at origin: f[k] = M for all k
        expected = jnp.full(N, M, dtype=jnp.complex64)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_points_at_pi(self):
        """Test with all points at x=pi."""
        M, N = 10, 32
        x = jnp.full(M, jnp.pi)
        c = jnp.ones(M, dtype=jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        # f[k] = M * exp(i*k*pi) = M * (-1)^k
        k = jnp.arange(-N // 2, (N + 1) // 2)
        expected = M * jnp.exp(1j * k * jnp.pi)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_regularly_spaced_points(self):
        """Test with regularly spaced points (should approach DFT)."""
        N = 32
        M = N  # Same number of points as modes
        x = jnp.linspace(-jnp.pi, jnp.pi, M, endpoint=False)
        # Delta function in frequency: only f[0] = M
        c = jnp.ones(M, dtype=jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        # Sum of evenly spaced exp(i*k*x) should be M for k=0, 0 otherwise
        # (due to orthogonality of complex exponentials)
        np.testing.assert_allclose(jnp.abs(result[N // 2]), M, rtol=1e-3)

    def test_negative_coordinates(self):
        """Test with negative coordinates in [-pi, 0)."""
        M, N = 50, 32
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=0.0)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.shape == (N,)
        assert not jnp.any(jnp.isnan(result))

    def test_coordinates_outside_standard_range(self):
        """Test with coordinates outside [-pi, pi) (should wrap)."""
        M, N = 50, 32
        key = jax.random.PRNGKey(42)
        # Coordinates in [pi, 3*pi)
        x = jax.random.uniform(key, (M,), minval=jnp.pi, maxval=3 * jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)

        # Should be same as wrapped coordinates
        x_wrapped = jnp.mod(x + jnp.pi, 2 * jnp.pi) - jnp.pi
        result_wrapped = nufft1d1(x_wrapped, c, N, eps=1e-6)

        np.testing.assert_allclose(result, result_wrapped, rtol=1e-5)


class TestSpecialStrengths:
    """Test with special strength values."""

    def test_zero_strengths(self):
        """Test with all zero strengths."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jnp.zeros(M, dtype=jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        np.testing.assert_allclose(result, jnp.zeros(N, dtype=jnp.complex64), atol=1e-10)

    def test_single_nonzero_strength(self):
        """Test with only one nonzero strength."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jnp.zeros(M, dtype=jnp.complex64)
        c = c.at[0].set(1.0 + 2j)

        result = nufft1d1(x, c, N, eps=1e-6)

        # Should equal result with just one point
        result_single = nufft1d1(x[:1], c[:1], N, eps=1e-6)
        np.testing.assert_allclose(result, result_single, rtol=1e-5)

    def test_purely_real_strengths(self):
        """Test with purely real strengths."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)).astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        # With real c and symmetric x distribution, should have conjugate symmetry
        # f[-k] = conj(f[k]) when c is real (approximately)
        assert not jnp.any(jnp.isnan(result))

    def test_purely_imaginary_strengths(self):
        """Test with purely imaginary strengths."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = 1j * jax.random.normal(jax.random.PRNGKey(43), (M,)).astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert not jnp.any(jnp.isnan(result))


class TestDifferentSizes:
    """Test with different problem sizes."""

    @pytest.mark.parametrize("N", [8, 16, 32, 64, 128, 256])
    def test_different_output_sizes(self, N):
        """Test with different numbers of modes."""
        M = 100
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.shape == (N,)
        assert not jnp.any(jnp.isnan(result))

    @pytest.mark.parametrize("M", [10, 100, 1000, 10000])
    def test_different_input_sizes(self, M):
        """Test with different numbers of nonuniform points."""
        N = 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.shape == (N,)
        assert not jnp.any(jnp.isnan(result))

    def test_more_points_than_modes(self):
        """Test M >> N case."""
        M, N = 10000, 16
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.shape == (N,)

    def test_more_modes_than_points(self):
        """Test N >> M case."""
        M, N = 10, 256
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.shape == (N,)


class TestTolerances:
    """Test different tolerance levels."""

    @pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
    def test_different_tolerances(self, eps):
        """Test that different tolerances work."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=eps)
        assert result.shape == (N,)
        assert not jnp.any(jnp.isnan(result))

    def test_tolerance_affects_accuracy(self):
        """Test that tighter tolerance gives more accurate results."""
        M, N = 50, 32  # Small for direct computation
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        # Direct computation
        k = jnp.arange(-N // 2, (N + 1) // 2)
        expected = jnp.sum(c[:, None] * jnp.exp(1j * k[None, :] * x[:, None]), axis=0)

        result_loose = nufft1d1(x, c, N, eps=1e-2)
        result_tight = nufft1d1(x, c, N, eps=1e-6)

        error_loose = jnp.linalg.norm(result_loose - expected) / jnp.linalg.norm(expected)
        error_tight = jnp.linalg.norm(result_tight - expected) / jnp.linalg.norm(expected)

        # Tighter tolerance should give smaller error (or both very small)
        assert error_tight < error_loose or (error_tight < 1e-4 and error_loose < 1e-1)


class TestModeOrdering:
    """Test mode ordering behavior."""

    def test_default_mode_ordering(self):
        """Test that default mode ordering produces valid output."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.shape == (N,)
        assert not jnp.any(jnp.isnan(result))


class TestSignConventions:
    """Test different sign conventions (isign)."""

    def test_isign_positive_vs_negative(self):
        """Test that isign=+1 and isign=-1 give valid results."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result_pos = nufft1d1(x, c, N, eps=1e-6, isign=1)  # exp(+ikx)
        result_neg = nufft1d1(x, c, N, eps=1e-6, isign=-1)  # exp(-ikx)

        # Both should produce valid output
        assert result_pos.shape == (N,)
        assert result_neg.shape == (N,)
        assert not jnp.any(jnp.isnan(result_pos))
        assert not jnp.any(jnp.isnan(result_neg))

        # They should be different (unless c or x is all zeros)
        assert not jnp.allclose(result_pos, result_neg)


class TestMultidimensional:
    """Test 2D and 3D edge cases."""

    def test_2d_single_point(self):
        """Test 2D NUFFT with single point."""
        x = jnp.array([0.5])
        y = jnp.array([0.3])
        c = jnp.array([1.0 + 1j])
        N1, N2 = 8, 8

        result = nufft2d1(x, y, c, (N1, N2), eps=1e-6)
        assert result.shape == (N1, N2)
        assert not jnp.any(jnp.isnan(result))

    def test_2d_points_on_axes(self):
        """Test 2D NUFFT with points on coordinate axes."""
        M, N1, N2 = 50, 16, 16
        key = jax.random.PRNGKey(42)

        # Points on x-axis (y=0)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jnp.zeros(M)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft2d1(x, y, c, (N1, N2), eps=1e-6)
        assert result.shape == (N1, N2)

    def test_3d_single_point(self):
        """Test 3D NUFFT with single point."""
        x = jnp.array([0.5])
        y = jnp.array([0.3])
        z = jnp.array([0.1])
        c = jnp.array([1.0 + 1j])
        N1, N2, N3 = 4, 4, 4

        result = nufft3d1(x, y, z, c, (N1, N2, N3), eps=1e-6)
        assert result.shape == (N1, N2, N3)
        assert not jnp.any(jnp.isnan(result))


class TestNumericalStability:
    """Test numerical stability with challenging inputs."""

    def test_large_coordinates(self):
        """Test with coordinates at extreme values."""
        M, N = 100, 64
        # Values very close to pi and -pi
        x = jnp.concatenate([jnp.full(M // 2, jnp.pi - 1e-6), jnp.full(M // 2, -jnp.pi + 1e-6)])
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))

    def test_very_small_strengths(self):
        """Test with very small strength values."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = (
            jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        ) * 1e-15
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert not jnp.any(jnp.isnan(result))
        # Result should be close to zero
        np.testing.assert_allclose(result, jnp.zeros_like(result), atol=1e-12)

    def test_very_large_strengths(self):
        """Test with very large strength values."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = (
            jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        ) * 1e10
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))
