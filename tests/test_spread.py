"""
Tests for NUFFT spreading and interpolation functions.

Tests spread_1d, spread_2d, spread_3d (Type 1 spreading)
and interp_1d, interp_2d, interp_3d (Type 2 interpolation).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import spreading functions when implemented
# from nufftax.core.spread import (
#     spread_1d, spread_2d, spread_3d,
#     interp_1d, interp_2d, interp_3d,
# )
from nufftax.core.kernel import compute_kernel_params

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def kernel_params():
    """Default kernel parameters for testing."""
    return compute_kernel_params(1e-6)


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


# ============================================================================
# Helper: Reference Spreading Implementation (Direct)
# ============================================================================


def spread_1d_direct(
    x: np.ndarray,
    c: np.ndarray,
    nf: int,
    nspread: int,
    beta: float,
    c_param: float,
) -> np.ndarray:
    """
    Direct (slow) implementation of 1D spreading for testing.

    For each nonuniform point x[j], add c[j] * kernel(grid - x[j])
    to the output grid.

    Args:
        x: Nonuniform points in [0, 2*pi), shape (M,)
        c: Complex strengths, shape (M,)
        nf: Fine grid size
        nspread: Kernel width
        beta: Kernel beta parameter
        c_param: Kernel c parameter

    Returns:
        fw: Complex array on fine grid, shape (nf,)
    """
    fw = np.zeros(nf, dtype=np.complex128)
    h = 2.0 * np.pi / nf  # Grid spacing

    for j in range(len(x)):
        # Map x[j] to grid index
        xi = x[j] / h  # Fractional grid index

        # Determine spreading range
        i0 = int(np.floor(xi - nspread / 2))

        for di in range(nspread):
            i = (i0 + di) % nf  # Wrap around
            # Distance from grid point to x[j]
            z = i0 + di - xi
            ker = np.exp(beta * (np.sqrt(max(0, 1 - c_param * z * z)) - 1))
            if 1 - c_param * z * z < 0:
                ker = 0.0
            fw[i] += c[j] * ker

    return fw


def interp_1d_direct(
    x: np.ndarray,
    fw: np.ndarray,
    nspread: int,
    beta: float,
    c_param: float,
) -> np.ndarray:
    """
    Direct (slow) implementation of 1D interpolation for testing.

    For each nonuniform point x[j], interpolate the grid values
    using the spreading kernel.

    Args:
        x: Nonuniform points in [0, 2*pi), shape (M,)
        fw: Complex array on fine grid, shape (nf,)
        nspread: Kernel width
        beta: Kernel beta parameter
        c_param: Kernel c parameter

    Returns:
        c: Complex values at nonuniform points, shape (M,)
    """
    M = len(x)
    nf = len(fw)
    c = np.zeros(M, dtype=np.complex128)
    h = 2.0 * np.pi / nf  # Grid spacing

    for j in range(M):
        # Map x[j] to grid index
        xi = x[j] / h  # Fractional grid index

        # Determine interpolation range
        i0 = int(np.floor(xi - nspread / 2))

        for di in range(nspread):
            i = (i0 + di) % nf  # Wrap around
            # Distance from grid point to x[j]
            z = i0 + di - xi
            ker = np.exp(beta * (np.sqrt(max(0, 1 - c_param * z * z)) - 1))
            if 1 - c_param * z * z < 0:
                ker = 0.0
            c[j] += fw[i] * ker

    return c


# ============================================================================
# Test Spreading Basic Properties
# ============================================================================


class TestSpreadBasic:
    """Basic tests for spreading operation."""

    def test_spread_1d_output_shape(self, kernel_params, rng):
        """Spreading should produce correct output shape."""
        M = 100
        rng.uniform(0, 2 * np.pi, M)
        rng.standard_normal(M) + 1j * rng.standard_normal(M)

        # fw = spread_1d(jnp.array(x), jnp.array(c), nf, kernel_params)
        # assert fw.shape == (nf,)

    def test_spread_1d_dtype_complex128(self, kernel_params, rng):
        """Output dtype should match input dtype."""
        M = 50
        rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        # fw = spread_1d(jnp.array(x), jnp.array(c), nf, kernel_params)
        # assert fw.dtype == jnp.complex128

    def test_spread_1d_dtype_complex64(self, kernel_params, rng):
        """Should support float32/complex64."""
        M = 50
        rng.uniform(0, 2 * np.pi, M).astype(np.float32)
        (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex64)

        # fw = spread_1d(jnp.array(x), jnp.array(c), nf, kernel_params)
        # assert fw.dtype == jnp.complex64


# ============================================================================
# Test Spreading Against Direct Implementation
# ============================================================================


class TestSpreadCorrectness:
    """Test spreading correctness against direct implementation."""

    def test_spread_1d_direct_single_point(self, kernel_params):
        """Test direct spread with single point."""
        x = np.array([np.pi])  # Center point
        c = np.array([1.0 + 0j])
        nf = 32
        nspread = kernel_params.nspread
        beta = kernel_params.beta
        c_param = kernel_params.c

        fw = spread_1d_direct(x, c, nf, nspread, beta, c_param)

        # Should have non-zero values near the center
        center_idx = nf // 2
        assert np.abs(fw[center_idx]) > 0

        # Sum should be approximately the strength times kernel integral
        # (normalized by grid spacing)
        assert np.sum(np.abs(fw)) > 0

    def test_spread_1d_direct_multiple_points(self, kernel_params, rng):
        """Test direct spread with multiple points."""
        M = 20
        nf = 64
        x = rng.uniform(0, 2 * np.pi, M)
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)

        nspread = kernel_params.nspread
        beta = kernel_params.beta
        c_param = kernel_params.c

        fw = spread_1d_direct(x, c, nf, nspread, beta, c_param)

        # Should be a complex array
        assert fw.dtype == np.complex128
        assert fw.shape == (nf,)

        # Should have non-trivial values
        assert np.linalg.norm(fw) > 0

    def test_spread_1d_vs_direct(self, kernel_params, rng):
        """Compare optimized spread_1d to direct implementation."""
        M = 50
        nf = 64
        x = rng.uniform(0, 2 * np.pi, M)
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)

        nspread = kernel_params.nspread
        beta = kernel_params.beta
        c_param = kernel_params.c

        # Direct implementation
        spread_1d_direct(x, c, nf, nspread, beta, c_param)

        # Optimized implementation
        # fw_opt = spread_1d(jnp.array(x), jnp.array(c), nf, kernel_params)

        # assert_allclose(np.array(fw_opt), fw_direct, rtol=1e-10)


# ============================================================================
# Test Interpolation Basic Properties
# ============================================================================


class TestInterpBasic:
    """Basic tests for interpolation operation."""

    def test_interp_1d_output_shape(self, kernel_params, rng):
        """Interpolation should produce correct output shape."""
        M = 100
        nf = 128
        rng.uniform(0, 2 * np.pi, M)
        rng.standard_normal(nf) + 1j * rng.standard_normal(nf)

        # c = interp_1d(jnp.array(x), jnp.array(fw), kernel_params)
        # assert c.shape == (M,)

    def test_interp_1d_dtype(self, kernel_params, rng):
        """Output dtype should match input dtype."""
        M = 50
        nf = 64
        rng.uniform(0, 2 * np.pi, M).astype(np.float64)
        (rng.standard_normal(nf) + 1j * rng.standard_normal(nf)).astype(np.complex128)

        # c = interp_1d(jnp.array(x), jnp.array(fw), kernel_params)
        # assert c.dtype == jnp.complex128


# ============================================================================
# Test Interpolation Correctness
# ============================================================================


class TestInterpCorrectness:
    """Test interpolation correctness against direct implementation."""

    def test_interp_1d_direct_single_point(self, kernel_params, rng):
        """Test direct interp with single point."""
        x = np.array([np.pi])  # Center point
        nf = 32
        fw = rng.standard_normal(nf) + 1j * rng.standard_normal(nf)

        nspread = kernel_params.nspread
        beta = kernel_params.beta
        c_param = kernel_params.c

        c = interp_1d_direct(x, fw, nspread, beta, c_param)

        # Should produce a single value
        assert c.shape == (1,)
        assert np.isfinite(c[0])

    def test_interp_1d_direct_constant_field(self, kernel_params):
        """Interpolating a constant should give (approximately) that constant."""
        M = 10
        nf = 64
        x = np.linspace(0, 2 * np.pi, M, endpoint=False)
        fw = np.ones(nf, dtype=np.complex128)

        nspread = kernel_params.nspread
        beta = kernel_params.beta
        c_param = kernel_params.c

        c = interp_1d_direct(x, fw, nspread, beta, c_param)

        # All values should be similar (kernel integral)
        # They won't be exactly 1 because the kernel doesn't integrate to 1
        assert np.std(np.abs(c)) / np.mean(np.abs(c)) < 0.1

    def test_interp_1d_vs_direct(self, kernel_params, rng):
        """Compare optimized interp_1d to direct implementation."""
        M = 50
        nf = 64
        x = rng.uniform(0, 2 * np.pi, M)
        fw = rng.standard_normal(nf) + 1j * rng.standard_normal(nf)

        nspread = kernel_params.nspread
        beta = kernel_params.beta
        c_param = kernel_params.c

        # Direct implementation
        interp_1d_direct(x, fw, nspread, beta, c_param)

        # Optimized implementation
        # c_opt = interp_1d(jnp.array(x), jnp.array(fw), kernel_params)

        # assert_allclose(np.array(c_opt), c_direct, rtol=1e-10)


# ============================================================================
# Test Spread-Interp Adjointness
# ============================================================================


class TestSpreadInterpAdjoint:
    """Test that spreading and interpolation are adjoints."""

    def test_adjoint_direct(self, kernel_params, rng):
        """Test adjoint property: <spread(c), fw> = <c, interp(fw)>."""
        M = 30
        nf = 48
        x = rng.uniform(0, 2 * np.pi, M)
        c = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        fw = rng.standard_normal(nf) + 1j * rng.standard_normal(nf)

        nspread = kernel_params.nspread
        beta = kernel_params.beta
        c_param = kernel_params.c

        # Forward: spread c to grid
        spread_c = spread_1d_direct(x, c, nf, nspread, beta, c_param)

        # Adjoint: interpolate fw to points
        interp_fw = interp_1d_direct(x, fw, nspread, beta, c_param)

        # Check adjoint: <spread(c), fw> == <c, interp(fw)>
        lhs = np.vdot(spread_c, fw)  # <spread(c), fw>
        rhs = np.vdot(c, interp_fw)  # <c, interp(fw)>

        assert_allclose(lhs, rhs, rtol=1e-10)

    def test_adjoint_optimized(self, kernel_params, rng):
        """Test adjoint property for optimized implementations."""
        M = 50
        nf = 64
        rng.uniform(0, 2 * np.pi, M)
        rng.standard_normal(M) + 1j * rng.standard_normal(M)
        rng.standard_normal(nf) + 1j * rng.standard_normal(nf)

        # Forward
        # spread_c = spread_1d(jnp.array(x), jnp.array(c), nf, kernel_params)

        # Adjoint
        # interp_fw = interp_1d(jnp.array(x), jnp.array(fw), kernel_params)

        # lhs = jnp.vdot(spread_c, jnp.array(fw))
        # rhs = jnp.vdot(jnp.array(c), interp_fw)

        # assert jnp.abs(lhs - rhs) / jnp.abs(lhs) < 1e-10


# ============================================================================
# Test JAX Transformations on Spreading
# ============================================================================


class TestSpreadJAXTransforms:
    """Test JAX transforms on spreading operations."""

    def test_spread_jit(self, kernel_params, rng):
        """Spreading should be JIT-compilable."""
        M = 50
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # @jax.jit
        # def compute(x, c):
        #     return spread_1d(x, c, nf, kernel_params)

        # result1 = compute(x, c)
        # result2 = compute(x, c)
        # assert_allclose(result1, result2, rtol=1e-14)

    def test_spread_grad_wrt_c(self, kernel_params, rng):
        """Should be able to differentiate spreading w.r.t. strengths."""
        M = 20
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # def loss(c):
        #     fw = spread_1d(x, c, nf, kernel_params)
        #     return jnp.sum(jnp.abs(fw) ** 2).real

        # grad = jax.grad(loss)(c)
        # assert grad.shape == c.shape
        # assert jnp.all(jnp.isfinite(grad))

    def test_spread_grad_wrt_x(self, kernel_params, rng):
        """Should be able to differentiate spreading w.r.t. positions."""
        M = 20
        jnp.array(rng.uniform(0.1, 2 * np.pi - 0.1, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # def loss(x):
        #     fw = spread_1d(x, c, nf, kernel_params)
        #     return jnp.sum(jnp.abs(fw) ** 2).real

        # grad = jax.grad(loss)(x)
        # assert grad.shape == x.shape
        # assert jnp.all(jnp.isfinite(grad))

    def test_spread_vmap(self, kernel_params, rng):
        """Spreading should support vmap over batches."""
        batch_size = 5
        M = 30

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        # batched_spread = jax.vmap(lambda c: spread_1d(x, c, nf, kernel_params))
        # fw_batch = batched_spread(c_batch)

        # assert fw_batch.shape == (batch_size, nf)


# ============================================================================
# Test 2D Spreading
# ============================================================================


class TestSpread2D:
    """Tests for 2D spreading and interpolation."""

    def test_spread_2d_shape(self, kernel_params, rng):
        """2D spreading should produce correct output shape."""
        M = 100
        _nf1, _nf2 = 32, 48

        rng.uniform(0, 2 * np.pi, M)
        rng.uniform(0, 2 * np.pi, M)
        rng.standard_normal(M) + 1j * rng.standard_normal(M)

        # fw = spread_2d(jnp.array(x), jnp.array(y), jnp.array(c),
        #                (nf1, nf2), kernel_params)
        # assert fw.shape == (nf1, nf2)

    def test_spread_interp_adjoint_2d(self, kernel_params, rng):
        """Test adjoint property in 2D."""
        M = 50
        nf1, nf2 = 32, 32

        rng.uniform(0, 2 * np.pi, M)
        rng.uniform(0, 2 * np.pi, M)
        rng.standard_normal(M) + 1j * rng.standard_normal(M)
        rng.standard_normal((nf1, nf2)) + 1j * rng.standard_normal((nf1, nf2))

        # spread_c = spread_2d(jnp.array(x), jnp.array(y), jnp.array(c),
        #                      (nf1, nf2), kernel_params)
        # interp_fw = interp_2d(jnp.array(x), jnp.array(y), jnp.array(fw),
        #                       kernel_params)

        # lhs = jnp.vdot(spread_c, jnp.array(fw))
        # rhs = jnp.vdot(jnp.array(c), interp_fw)

        # assert jnp.abs(lhs - rhs) / jnp.abs(lhs) < 1e-10


# ============================================================================
# Test 3D Spreading
# ============================================================================


class TestSpread3D:
    """Tests for 3D spreading and interpolation."""

    def test_spread_3d_shape(self, kernel_params, rng):
        """3D spreading should produce correct output shape."""
        M = 100
        _nf1, _nf2, _nf3 = 16, 16, 16

        rng.uniform(0, 2 * np.pi, M)
        rng.uniform(0, 2 * np.pi, M)
        rng.uniform(0, 2 * np.pi, M)
        rng.standard_normal(M) + 1j * rng.standard_normal(M)

        # fw = spread_3d(jnp.array(x), jnp.array(y), jnp.array(z),
        #                jnp.array(c), (nf1, nf2, nf3), kernel_params)
        # assert fw.shape == (nf1, nf2, nf3)

    def test_spread_interp_adjoint_3d(self, kernel_params, rng):
        """Test adjoint property in 3D."""
        M = 30
        nf1, nf2, nf3 = 12, 12, 12

        rng.uniform(0, 2 * np.pi, M)
        rng.uniform(0, 2 * np.pi, M)
        rng.uniform(0, 2 * np.pi, M)
        rng.standard_normal(M) + 1j * rng.standard_normal(M)
        (rng.standard_normal((nf1, nf2, nf3)) + 1j * rng.standard_normal((nf1, nf2, nf3)))

        # spread_c = spread_3d(..., kernel_params)
        # interp_fw = interp_3d(..., kernel_params)

        # lhs = jnp.vdot(spread_c, jnp.array(fw))
        # rhs = jnp.vdot(jnp.array(c), interp_fw)

        # assert jnp.abs(lhs - rhs) / jnp.abs(lhs) < 1e-10


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestSpreadEdgeCases:
    """Test edge cases for spreading/interpolation."""

    def test_spread_single_point(self, kernel_params):
        """Spreading with a single point."""
        jnp.array([np.pi])
        jnp.array([1.0 + 0j])

        # fw = spread_1d(x, c, nf, kernel_params)
        # assert fw.shape == (nf,)
        # assert jnp.any(jnp.abs(fw) > 0)

    def test_spread_points_at_boundary(self, kernel_params, rng):
        """Spreading with points near domain boundaries."""
        # Points very close to 0 and 2*pi
        jnp.array([1e-10, 2 * np.pi - 1e-10])
        jnp.array([1.0 + 0j, 1.0 + 0j])

        # fw = spread_1d(x, c, nf, kernel_params)
        # Should handle wrapping correctly
        # assert jnp.all(jnp.isfinite(fw))

    def test_spread_zero_strengths(self, kernel_params, rng):
        """Spreading with zero strengths should give zero output."""
        M = 50
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.zeros(M, dtype=jnp.complex64)

        # fw = spread_1d(x, c, nf, kernel_params)
        # assert_allclose(fw, 0.0, atol=1e-15)

    def test_interp_zero_field(self, kernel_params, rng):
        """Interpolating zero field should give zero."""
        M = 50
        nf = 64
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.zeros(nf, dtype=jnp.complex64)

        # c = interp_1d(x, fw, kernel_params)
        # assert_allclose(c, 0.0, atol=1e-15)


# ============================================================================
# Test Performance Characteristics
# ============================================================================


class TestSpreadPerformance:
    """Test performance-related aspects of spreading."""

    @pytest.mark.slow
    def test_spread_large_problem(self, kernel_params, rng):
        """Test spreading with large problem size."""
        M = 100000

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # Should complete without memory issues
        # fw = spread_1d(x, c, nf, kernel_params)
        # assert fw.shape == (nf,)

    def test_spread_repeated_compilation(self, kernel_params, rng):
        """JIT should not recompile for same shapes."""
        M = 100

        # @jax.jit
        # def compute(x, c):
        #     return spread_1d(x, c, nf, kernel_params)

        # First call triggers compilation
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        # _ = compute(x1, c1)

        # Second call with same shapes should use cached compilation
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        # _ = compute(x2, c2)
