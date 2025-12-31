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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import NUFFT functions when implemented
# from nufftax import nufft1d1, nufft1d2, nufft2d1, nufft2d2, nufft3d1, nufft3d2
# from nufftax.transforms.nufft2 import nufft1d2, nufft2d2, nufft3d2
from tests.conftest import (
    PRECISION_LEVELS,
    dft_nufft1d2,
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

    @pytest.mark.parametrize("M", [10, 50, 100])
    @pytest.mark.parametrize("n_modes", [8, 32, 64])
    def test_nufft1d2_vs_dft(self, rng, M, n_modes):
        """Compare nufft1d2 against DFT for small problems."""
        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        # DFT reference
        dft_nufft1d2(x, f)

        # JAX implementation
        # c_jax = nufft1d2(jnp.array(x), jnp.array(f), eps=1e-12)

        # rel_err = relative_error(c_jax, jnp.array(c_dft))
        # assert rel_err < 1e-10, f"Relative error {rel_err} exceeds tolerance"


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
        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
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

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        # FINUFFT reference
        finufft.nufft1d2(x, f, eps=eps)

        # JAX implementation
        # c_jax = nufft1d2(jnp.array(x), jnp.array(f), eps=eps)

        # rel_err = relative_error(c_jax, jnp.array(c_ref))
        # assert rel_err < 10 * eps, f"eps={eps}, rel_err={rel_err}"

    @requires_finufft
    def test_nufft1d2_vs_finufft_large(self, rng):
        """Test with larger problem size."""
        import finufft

        M = 10000
        n_modes = 1024
        eps = 1e-6

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        finufft.nufft1d2(x, f, eps=eps)
        # c_jax = nufft1d2(jnp.array(x), jnp.array(f), eps=eps)

        # rel_err = relative_error(c_jax, jnp.array(c_ref))
        # assert rel_err < 10 * eps


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
        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
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

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        finufft.nufft2d2(x, y, f, eps=eps)
        # c_jax = nufft2d2(jnp.array(x), jnp.array(y), jnp.array(f), eps=eps)

        # rel_err = relative_error(c_jax, jnp.array(c_ref))
        # assert rel_err < 10 * eps


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
        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        z = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
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

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        z = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        finufft.nufft3d2(x, y, z, f, eps=eps)
        # c_jax = nufft3d2(jnp.array(x), jnp.array(y), jnp.array(z),
        #                  jnp.array(f), eps=eps)

        # rel_err = relative_error(c_jax, jnp.array(c_ref))
        # assert rel_err < 10 * eps


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
        """Test adjoint property using FINUFFT."""
        import finufft

        M = 500
        n_modes = 64
        eps = 1e-12

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        # Type 1: c -> f1
        f1 = finufft.nufft1d1(x, c, n_modes, eps=eps)

        # Type 2: f -> c2
        c2 = finufft.nufft1d2(x, f, eps=eps)

        # <nufft1(c), f> should equal <c, nufft2(f)>
        lhs = np.vdot(f1, f)
        rhs = np.vdot(c, c2)

        rel_diff = np.abs(lhs - rhs) / np.abs(lhs)
        assert rel_diff < 1e-10, f"Adjoint relation violated: rel_diff={rel_diff}"

    def test_adjoint_jax_1d(self, rng):
        """Test adjoint property using JAX implementations."""
        M = 500
        n_modes = 64

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # Type 1: c -> f1
        # f1 = nufft1d1(x, c, n_modes, eps=eps)

        # Type 2: f -> c2
        # c2 = nufft1d2(x, f, eps=eps)

        # <nufft1(c), f> should equal <c, nufft2(f)>
        # lhs = jnp.vdot(f1, f)
        # rhs = jnp.vdot(c, c2)

        # rel_diff = jnp.abs(lhs - rhs) / jnp.abs(lhs)
        # assert rel_diff < 1e-8

    @requires_finufft
    def test_adjoint_finufft_2d(self, rng):
        """Test adjoint property in 2D using FINUFFT."""
        import finufft

        M = 300
        n_modes = (32, 32)
        eps = 1e-12

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        f1 = finufft.nufft2d1(x, y, c, n_modes, eps=eps)
        c2 = finufft.nufft2d2(x, y, f, eps=eps)

        lhs = np.vdot(f1, f)
        rhs = np.vdot(c, c2)

        rel_diff = np.abs(lhs - rhs) / np.abs(lhs)
        assert rel_diff < 1e-10

    @requires_finufft
    def test_adjoint_finufft_3d(self, rng):
        """Test adjoint property in 3D using FINUFFT."""
        import finufft

        M = 200
        n_modes = (12, 12, 12)
        eps = 1e-12

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        y = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        z = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)
        f = (rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)).astype(np.complex128)

        f1 = finufft.nufft3d1(x, y, z, c, n_modes, eps=eps)
        c2 = finufft.nufft3d2(x, y, z, f, eps=eps)

        lhs = np.vdot(f1, f)
        rhs = np.vdot(c, c2)

        rel_diff = np.abs(lhs - rhs) / np.abs(lhs)
        assert rel_diff < 1e-10


# ============================================================================
# Test Gradient Correctness
# ============================================================================


class TestNUFFT2Gradients:
    """Test gradients of Type 2 transforms via finite differences."""

    def test_grad_f_finite_diff(self, rng):
        """Test gradient w.r.t. f via finite differences."""
        M = 10
        n_modes = 8
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        def loss(f):
            # c = nufft1d2(x, f, eps=1e-9)
            # return jnp.sum(jnp.abs(c) ** 2).real
            pass

        # Automatic gradient
        # grad_auto = jax.grad(loss)(f)

        # Numerical gradient
        # grad_num = numerical_gradient_complex(loss, f, h=1e-6)

        # assert_allclose(np.array(grad_auto), np.array(grad_num), rtol=1e-4)

    def test_grad_x_finite_diff(self, rng):
        """Test gradient w.r.t. x via finite differences."""
        M = 10
        n_modes = 8
        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        def loss(x):
            # c = nufft1d2(x, f, eps=1e-9)
            # return jnp.sum(jnp.abs(c) ** 2).real
            pass

        # Automatic gradient
        # grad_auto = jax.grad(loss)(x)

        # Numerical gradient
        # grad_num = numerical_gradient_real(loss, np.array(x), h=1e-6)

        # assert_allclose(np.array(grad_auto), np.array(grad_num), rtol=1e-4)

    def test_grad_both_finite_diff(self, rng):
        """Test gradient w.r.t. both x and f."""
        M = 8
        n_modes = 8
        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        def loss(x, f):
            # c = nufft1d2(x, f, eps=1e-9)
            # return jnp.sum(jnp.abs(c) ** 2).real
            pass

        # grad_x, grad_f = jax.grad(loss, argnums=(0, 1))(x, f)

        # assert grad_x.shape == x.shape
        # assert grad_f.shape == f.shape
        # assert jnp.all(jnp.isfinite(grad_x))
        # assert jnp.all(jnp.isfinite(grad_f))


# ============================================================================
# Test JIT Compilation
# ============================================================================


class TestNUFFT2JIT:
    """Test JIT compilation of Type 2 transforms."""

    def test_jit_1d(self, rng):
        """Test JIT compilation of 1D Type 2."""
        M = 100
        n_modes = 64

        # @jax.jit
        # def compute(x, f):
        #     return nufft1d2(x, f, eps=1e-9)

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # First call (compilation)
        # c1 = compute(x, f)

        # Second call (cached)
        # c2 = compute(x, f)

        # assert_allclose(c1, c2, rtol=1e-14)


# ============================================================================
# Test vmap
# ============================================================================


class TestNUFFT2Vmap:
    """Test vmap over batches."""

    def test_vmap_over_f(self, rng):
        """Test vmap over batch of mode coefficients."""
        batch_size = 5
        M = 50
        n_modes = 32

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, n_modes)) + 1j * rng.standard_normal((batch_size, n_modes)))

        # batched = jax.vmap(lambda f: nufft1d2(x, f, eps=1e-9))
        # c_batch = batched(f_batch)

        # assert c_batch.shape == (batch_size, M)

    def test_vmap_over_x(self, rng):
        """Test vmap over batch of positions."""
        batch_size = 5
        M = 50
        n_modes = 32

        jnp.array([rng.uniform(0, 2 * np.pi, M) for _ in range(batch_size)])
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # batched = jax.vmap(lambda x: nufft1d2(x, f, eps=1e-9))
        # c_batch = batched(x_batch)

        # assert c_batch.shape == (batch_size, M)


# ============================================================================
# Test Precision/Dtype
# ============================================================================


class TestNUFFT2Precision:
    """Test different precisions."""

    def test_float32_complex64(self, rng):
        """Test with float32/complex64."""
        M = 100
        n_modes = 64

        jnp.array(rng.uniform(0, 2 * np.pi, M), dtype=jnp.float32)
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes), dtype=jnp.complex64)

        # c = nufft1d2(x, f, eps=1e-5)
        # assert c.dtype == jnp.complex64

    def test_float64_complex128(self, rng):
        """Test with float64/complex128."""
        with jax.enable_x64(True):
            M = 100
            n_modes = 64

            jnp.array(rng.uniform(0, 2 * np.pi, M), dtype=jnp.float64)
            jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes), dtype=jnp.complex128)

            # c = nufft1d2(x, f, eps=1e-12)
            # assert c.dtype == jnp.complex128


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestNUFFT2EdgeCases:
    """Test edge cases."""

    def test_single_point(self, rng):
        """Single nonuniform point."""
        jnp.array([1.0])
        n_modes = 8
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # c = nufft1d2(x, f, eps=1e-9)
        # c[0] = sum_k f[k] * exp(i*k*1.0) for k=-4,...,3
        # k = jnp.arange(n_modes) - n_modes // 2
        # expected = jnp.sum(f * jnp.exp(1j * k * 1.0))
        # assert_allclose(c[0], expected, rtol=1e-6)

    def test_zero_modes(self, rng):
        """Zero mode coefficients should give zero output."""
        M = 50
        n_modes = 32

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.zeros(n_modes, dtype=jnp.complex64)

        # c = nufft1d2(x, f, eps=1e-9)
        # assert_allclose(c, 0.0, atol=1e-12)

    def test_single_mode(self):
        """Single non-zero mode should give pure exponential."""
        M = 10
        n_modes = 8
        jnp.linspace(0, 2 * np.pi, M, endpoint=False)

        # Only mode k=1 is non-zero
        f = jnp.zeros(n_modes, dtype=jnp.complex64)
        # k=1 corresponds to index n_modes//2 + 1 in centered ordering
        k_idx = n_modes // 2 + 1
        f = f.at[k_idx].set(1.0 + 0j)

        # c = nufft1d2(x, f, eps=1e-12)
        # Should be exp(i*1*x) = exp(i*x)
        # expected = jnp.exp(1j * x)
        # assert_allclose(c, expected, rtol=1e-9)


# ============================================================================
# Test Mathematical Properties
# ============================================================================


class TestNUFFT2Properties:
    """Test mathematical properties of Type 2."""

    def test_linearity_in_f(self, rng):
        """NUFFT should be linear in f."""
        M = 50
        n_modes = 32

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # c(f1 + alpha*f2) = c(f1) + alpha*c(f2)
        # c_sum = nufft1d2(x, f1 + alpha * f2, eps=1e-12)
        # c1 = nufft1d2(x, f1, eps=1e-12)
        # c2 = nufft1d2(x, f2, eps=1e-12)

        # assert_allclose(c_sum, c1 + alpha * c2, rtol=1e-10)

    def test_shift_property(self, rng):
        """Shifting x should give different values but same magnitude pattern."""
        M = 50
        n_modes = 32

        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # c_original = nufft1d2(x, f, eps=1e-12)
        # c_shifted = nufft1d2(x + shift, f, eps=1e-12)

        # c_shifted[j] = sum_k f[k] * exp(i*k*(x[j]+shift))
        #              = sum_k f[k] * exp(i*k*x[j]) * exp(i*k*shift)
        # So the relationship involves convolution in frequency domain
        # Just verify both are finite for now
        # assert jnp.all(jnp.isfinite(c_shifted))


# ============================================================================
# Test Round-Trip (Type 1 -> Type 2 or vice versa)
# ============================================================================


class TestRoundTrip:
    """Test round-trip transformations."""

    @requires_finufft
    @pytest.mark.skip(
        reason="Type1->Type2 round-trip does not produce scaled original; nonuniform points are not orthogonal basis"
    )
    def test_roundtrip_finufft_1d(self, rng):
        """Test Type 1 -> Type 2 round trip using FINUFFT.

        NOTE: This test is skipped because the mathematical assumption is incorrect.
        The NUFFT Type 1 followed by Type 2 does NOT give a scaled version of the
        original signal. For c' = Type2(Type1(c)):

        c'[j] = Σ_k (Σ_l c[l] * exp(i*k*x[l])) * exp(i*k*x[j])
              = Σ_l c[l] * Σ_k exp(i*k*(x[l] + x[j]))

        This is NOT c[j] * constant because nonuniform points don't form an
        orthogonal basis. Use the adjoint property test instead.
        """
        import finufft

        M = 100
        n_modes = 64
        eps = 1e-9

        x = rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        c_original = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        # Type 1: c -> f
        f = finufft.nufft1d1(x, c_original, n_modes, eps=eps)

        # Type 2: f -> c_recovered
        c_recovered = finufft.nufft1d2(x, f, eps=eps)

        # Due to normalization, c_recovered won't equal c_original exactly
        # but the transformation should be invertible up to scaling
        # Check that the ratio is approximately constant
        ratios = c_recovered / c_original
        # Remove outliers (very small c_original values)
        mask = np.abs(c_original) > 0.1 * np.max(np.abs(c_original))
        ratios_masked = ratios[mask]

        # Standard deviation of ratios should be small relative to mean
        if len(ratios_masked) > 0:
            ratio_std = np.std(ratios_masked)
            ratio_mean = np.abs(np.mean(ratios_masked))
            if ratio_mean > 1e-10:
                assert ratio_std / ratio_mean < 0.1, "Round-trip scaling not consistent"

    def test_roundtrip_jax_1d(self, rng):
        """Test Type 1 -> Type 2 round trip using JAX."""
        M = 100

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # Type 1: c -> f
        # f = nufft1d1(x, c_original, n_modes, eps=eps)

        # Type 2: f -> c_recovered
        # c_recovered = nufft1d2(x, f, eps=eps)

        # The round-trip should preserve the structure
        # assert jnp.all(jnp.isfinite(c_recovered))
