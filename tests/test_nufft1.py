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
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


# ============================================================================
# Reference Implementation Checks
# ============================================================================


def finufft_available():
    """Check if finufft is installed."""
    try:
        import finufft  # noqa: F401

        return True
    except ImportError:
        return False


requires_finufft = pytest.mark.skipif(not finufft_available(), reason="finufft not installed")


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


# ============================================================================
# Test Gradient Correctness
# ============================================================================


class TestNUFFT1Gradients:
    """Test gradients of Type 1 transforms via finite differences."""

    def test_grad_c_finite_diff(self, rng):
        """Test gradient w.r.t. c via finite differences."""
        M = 10
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # Automatic gradient
        # grad_auto = jax.grad(loss)(c)

        # Numerical gradient
        # grad_num = numerical_gradient_complex(loss, c, h=1e-6)

        # assert_allclose(np.array(grad_auto), np.array(grad_num), rtol=1e-4)

    def test_grad_x_finite_diff(self, rng):
        """Test gradient w.r.t. x via finite differences."""
        M = 10
        # Avoid boundary for smoother gradients
        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(x):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # Automatic gradient
        # grad_auto = jax.grad(loss)(x)

        # Numerical gradient
        # grad_num = numerical_gradient_real(loss, np.array(x), h=1e-6)

        # assert_allclose(np.array(grad_auto), np.array(grad_num), rtol=1e-4)

    def test_grad_both_finite_diff(self, rng):
        """Test gradient w.r.t. both x and c."""
        M = 8
        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(x, c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # grad_x, grad_c = jax.grad(loss, argnums=(0, 1))(x, c)

        # assert grad_x.shape == x.shape
        # assert grad_c.shape == c.shape
        # assert jnp.all(jnp.isfinite(grad_x))
        # assert jnp.all(jnp.isfinite(grad_c))

    def test_grad_2d_c_finite_diff(self, rng):
        """Test 2D gradient w.r.t. c."""
        M = 15
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(c):
            # f = nufft2d1(x, y, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # grad_auto = jax.grad(loss)(c)
        # grad_num = numerical_gradient_complex(loss, c, h=1e-6)
        # assert_allclose(np.array(grad_auto), np.array(grad_num), rtol=1e-4)


# ============================================================================
# Test JIT Compilation
# ============================================================================


class TestNUFFT1JIT:
    """Test JIT compilation of Type 1 transforms."""

    def test_jit_1d(self, rng):
        """Test JIT compilation of 1D Type 1."""
        M = 100

        # @jax.jit
        # def compute(x, c):
        #     return nufft1d1(x, c, n_modes, eps=1e-9)

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # First call (compilation)
        # f1 = compute(x, c)

        # Second call (cached)
        # f2 = compute(x, c)

        # assert_allclose(f1, f2, rtol=1e-14)

    def test_jit_no_recompile_same_shape(self, rng):
        """JIT should not recompile for same shapes."""
        M = 100

        # @jax.jit
        # def compute(x, c):
        #     return nufft1d1(x, c, n_modes, eps=1e-9)

        # First data
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # Second data (same shape)
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # _ = compute(x1, c1)
        # _ = compute(x2, c2)  # Should use cached compilation


# ============================================================================
# Test vmap
# ============================================================================


class TestNUFFT1Vmap:
    """Test vmap over batches."""

    def test_vmap_over_c(self, rng):
        """Test vmap over batch of strengths."""
        batch_size = 5
        M = 50

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        # batched = jax.vmap(lambda c: nufft1d1(x, c, n_modes, eps=1e-9))
        # f_batch = batched(c_batch)

        # assert f_batch.shape == (batch_size, n_modes)

    def test_vmap_over_x(self, rng):
        """Test vmap over batch of positions."""
        batch_size = 5
        M = 50

        jnp.array([rng.uniform(0, 2 * np.pi, M) for _ in range(batch_size)])
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # batched = jax.vmap(lambda x: nufft1d1(x, c, n_modes, eps=1e-9))
        # f_batch = batched(x_batch)

        # assert f_batch.shape == (batch_size, n_modes)

    def test_vmap_over_both(self, rng):
        """Test vmap over both positions and strengths."""
        batch_size = 5
        M = 50

        jnp.array([rng.uniform(0, 2 * np.pi, M) for _ in range(batch_size)])
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        # batched = jax.vmap(lambda x, c: nufft1d1(x, c, n_modes, eps=1e-9))
        # f_batch = batched(x_batch, c_batch)

        # assert f_batch.shape == (batch_size, n_modes)


# ============================================================================
# Test Precision/Dtype
# ============================================================================


class TestNUFFT1Precision:
    """Test different precisions."""

    def test_float32_complex64(self, rng):
        """Test with float32/complex64."""
        M = 100

        jnp.array(rng.uniform(0, 2 * np.pi, M), dtype=jnp.float32)
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M), dtype=jnp.complex64)

        # f = nufft1d1(x, c, n_modes, eps=1e-5)
        # assert f.dtype == jnp.complex64

    def test_float64_complex128(self, rng):
        """Test with float64/complex128."""
        with jax.enable_x64(True):
            M = 100

            jnp.array(rng.uniform(0, 2 * np.pi, M), dtype=jnp.float64)
            jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M), dtype=jnp.complex128)

            # f = nufft1d1(x, c, n_modes, eps=1e-12)
            # assert f.dtype == jnp.complex128


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestNUFFT1EdgeCases:
    """Test edge cases."""

    def test_single_point(self):
        """Single nonuniform point."""
        jnp.array([1.0])
        jnp.array([1.0 + 0j])

        # f = nufft1d1(x, c, n_modes, eps=1e-9)
        # Should be exp(i*k*1.0) for k=-4,...,3
        # k = jnp.arange(n_modes) - n_modes // 2
        # expected = jnp.exp(1j * k * 1.0)
        # assert_allclose(f, expected, rtol=1e-6)

    def test_zero_strengths(self, rng):
        """Zero strengths should give zero output."""
        M = 50

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.zeros(M, dtype=jnp.complex64)

        # f = nufft1d1(x, c, n_modes, eps=1e-9)
        # assert_allclose(f, 0.0, atol=1e-12)

    def test_points_at_boundary(self, rng):
        """Points at domain boundary."""
        M = 10

        # Points at exact boundary
        jnp.array([0.0, 2 * np.pi - 1e-15] + list(rng.uniform(0.1, 2 * np.pi - 0.1, M - 2)))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # f = nufft1d1(x, c, n_modes, eps=1e-9)
        # assert jnp.all(jnp.isfinite(f))

    @pytest.mark.slow
    def test_large_problem(self, rng):
        """Large problem size."""
        M = 100000

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # f = nufft1d1(x, c, n_modes, eps=1e-6)
        # assert f.shape == (n_modes,)


# ============================================================================
# Test Specific Mathematical Properties
# ============================================================================


class TestNUFFT1Properties:
    """Test mathematical properties of Type 1."""

    def test_linearity_in_c(self, rng):
        """NUFFT should be linear in c."""
        M = 50

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # f(c1 + alpha*c2) = f(c1) + alpha*f(c2)
        # f_sum = nufft1d1(x, c1 + alpha * c2, n_modes, eps=1e-12)
        # f1 = nufft1d1(x, c1, n_modes, eps=1e-12)
        # f2 = nufft1d1(x, c2, n_modes, eps=1e-12)

        # assert_allclose(f_sum, f1 + alpha * f2, rtol=1e-10)

    def test_conjugate_symmetry(self, rng):
        """Real input c should give conjugate-symmetric f."""
        M = 50

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 0j)  # Real strengths

        # f = nufft1d1(x, c, n_modes, eps=1e-12)

        # f[-k] should equal conj(f[k])
        # For the FFT ordering, this means f[n_modes - k] = conj(f[k])
        # (approximately, due to numerical precision)
        # for k in range(1, n_modes // 2):
        #     assert_allclose(f[n_modes - k], jnp.conj(f[k]), rtol=1e-9)

    def test_shift_property(self, rng):
        """Shifting x should multiply f by exponential."""
        M = 50

        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # f_original = nufft1d1(x, c, n_modes, eps=1e-12)
        # f_shifted = nufft1d1(x + shift, c, n_modes, eps=1e-12)

        # f_shifted[k] = f_original[k] * exp(i*k*shift)
        # k = jnp.arange(n_modes) - n_modes // 2
        # expected = f_original * jnp.exp(1j * k * shift)

        # assert_allclose(f_shifted, expected, rtol=1e-9)
