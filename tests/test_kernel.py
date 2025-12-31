"""
Tests for NUFFT kernel functions.

Tests the ES (Exponential of Semicircle) kernel evaluation,
kernel derivatives, parameter computation, and Fourier series.
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from nufftax.core.kernel import (
    KernelParams,
    compute_kernel_params,
    es_kernel,
    es_kernel_derivative,
    kernel_fourier_series,
)

# ============================================================================
# Test ES Kernel Evaluation
# ============================================================================


class TestESKernel:
    """Tests for es_kernel function."""

    def test_kernel_at_origin(self):
        """Kernel at z=0 should equal 1."""
        z = jnp.array([0.0])
        beta = 10.0
        c = 0.25  # nspread=4

        result = es_kernel(z, beta, c)
        assert_allclose(result, 1.0, rtol=1e-10)

    def test_kernel_symmetry(self):
        """Kernel should be symmetric: phi(-z) = phi(z)."""
        z = jnp.array([0.3, 0.7, 1.0, 1.5])
        beta = 12.0
        c = 0.25

        phi_pos = es_kernel(z, beta, c)
        phi_neg = es_kernel(-z, beta, c)

        assert_allclose(phi_pos, phi_neg, rtol=1e-10)

    def test_kernel_support(self):
        """Kernel should be zero outside support where c*z^2 > 1."""
        beta = 10.0
        nspread = 4
        c = 4.0 / (nspread**2)

        # z values at boundary and beyond
        z_in = jnp.array([0.0, 0.5, 1.0, 1.9, 1.99])
        z_out = jnp.array([2.1, 3.0, 5.0, 10.0])

        # Inside support: should be positive
        result_in = es_kernel(z_in, beta, c)
        assert jnp.all(result_in > 0), "Kernel should be positive inside support"

        # Outside support: should be zero
        result_out = es_kernel(z_out, beta, c)
        assert_allclose(result_out, 0.0, atol=1e-14)

    def test_kernel_monotonicity(self):
        """Kernel should be monotonically decreasing for z >= 0."""
        z = jnp.linspace(0, 1.8, 50)
        beta = 12.0
        c = 0.25

        phi = es_kernel(z, beta, c)

        # Check monotonicity: phi[i+1] <= phi[i] for all i
        diffs = phi[1:] - phi[:-1]
        assert jnp.all(diffs <= 1e-10), "Kernel should be monotonically decreasing"

    def test_kernel_vectorized(self):
        """Kernel should work on arbitrary array shapes."""
        beta = 10.0
        c = 0.25

        # 1D array
        z1 = jnp.linspace(-1, 1, 100)
        result1 = es_kernel(z1, beta, c)
        assert result1.shape == (100,)

        # 2D array
        z2 = jnp.linspace(-1, 1, 100).reshape(10, 10)
        result2 = es_kernel(z2, beta, c)
        assert result2.shape == (10, 10)

        # 3D array
        z3 = jnp.linspace(-1, 1, 24).reshape(2, 3, 4)
        result3 = es_kernel(z3, beta, c)
        assert result3.shape == (2, 3, 4)

    def test_kernel_jit(self):
        """Kernel should be JIT-compilable."""

        @jax.jit
        def compute_kernel(z):
            return es_kernel(z, 10.0, 0.25)

        z = jnp.linspace(-1, 1, 50)

        # First call (compilation)
        result1 = compute_kernel(z)
        # Second call (cached)
        result2 = compute_kernel(z)

        assert_allclose(result1, result2, rtol=1e-14)

    def test_kernel_grad(self):
        """Kernel should have valid gradients."""

        def loss(z):
            return jnp.sum(es_kernel(z, 10.0, 0.25) ** 2)

        z = jnp.array([0.0, 0.5, 1.0])
        grad = jax.grad(loss)(z)

        assert grad.shape == z.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_kernel_different_params(self):
        """Test kernel with various parameter combinations."""
        # Use odd number of points so z=0 is exactly at the center
        z = jnp.linspace(-1.5, 1.5, 51)

        for nspread in [2, 4, 6, 8, 10, 12]:
            c = 4.0 / (nspread**2)
            for beta in [5.0, 10.0, 15.0, 20.0]:
                phi = es_kernel(z, beta, c)
                # Should be non-negative
                assert jnp.all(phi >= -1e-14)
                # Max should be at or near origin (within 1 index for numerical reasons)
                max_idx = int(jnp.argmax(phi))
                center_idx = len(z) // 2
                assert abs(max_idx - center_idx) <= 1, f"Max at {max_idx}, expected near {center_idx}"


# ============================================================================
# Test ES Kernel Derivative
# ============================================================================


class TestESKernelDerivative:
    """Tests for es_kernel_derivative function."""

    def test_derivative_at_origin(self):
        """Derivative at z=0 should be 0 (extremum)."""
        z = jnp.array([0.0])
        beta = 10.0
        c = 0.25

        dphi = es_kernel_derivative(z, beta, c)
        assert_allclose(dphi, 0.0, atol=1e-10)

    def test_derivative_antisymmetry(self):
        """Derivative should be antisymmetric: phi'(-z) = -phi'(z)."""
        z = jnp.array([0.1, 0.5, 1.0, 1.5])
        beta = 12.0
        c = 0.25

        dphi_pos = es_kernel_derivative(z, beta, c)
        dphi_neg = es_kernel_derivative(-z, beta, c)

        assert_allclose(dphi_pos, -dphi_neg, rtol=1e-10)

    def test_derivative_numerical(self):
        """Compare derivative to finite differences."""
        # Use float64 for numerical precision in finite difference calculation
        # Enable x64 mode for this test
        with jax.enable_x64(True):
            z = jnp.array([0.3, 0.7, 1.2], dtype=jnp.float64)
            beta = 10.0
            c = 0.25
            h = 1e-6

            # Numerical derivative
            phi_plus = es_kernel(z + h, beta, c)
            phi_minus = es_kernel(z - h, beta, c)
            dphi_numerical = (phi_plus - phi_minus) / (2 * h)

            # Analytical derivative
            dphi_analytical = es_kernel_derivative(z, beta, c)

            assert_allclose(dphi_analytical, dphi_numerical, rtol=1e-5)

    def test_derivative_sign(self):
        """Derivative should be negative for z > 0 (kernel is decreasing)."""
        z = jnp.array([0.1, 0.5, 1.0, 1.5])
        beta = 10.0
        c = 0.25

        dphi = es_kernel_derivative(z, beta, c)
        # Where kernel is non-zero, derivative should be negative
        mask = es_kernel(z, beta, c) > 1e-10
        assert jnp.all(dphi[mask] <= 0)

    def test_derivative_jit(self):
        """Derivative should be JIT-compilable."""

        @jax.jit
        def compute_derivative(z):
            return es_kernel_derivative(z, 10.0, 0.25)

        z = jnp.linspace(0.1, 1.5, 20)
        result = compute_derivative(z)
        assert jnp.all(jnp.isfinite(result))


# ============================================================================
# Test Kernel Parameter Computation
# ============================================================================


class TestKernelParams:
    """Tests for compute_kernel_params function."""

    @pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    def test_params_tolerance_scaling(self, tol):
        """Higher precision should require larger nspread."""
        params = compute_kernel_params(tol)

        # nspread should be at least 2
        assert params.nspread >= 2

        # nspread should be at most max_nspread
        assert params.nspread <= 16

        # beta should be positive
        assert params.beta > 0

        # c should be 4/nspread^2
        expected_c = 4.0 / (params.nspread**2)
        assert_allclose(params.c, expected_c, rtol=1e-10)

    def test_params_nspread_increases_with_precision(self):
        """nspread should increase as tolerance decreases."""
        tols = [1e-2, 1e-6, 1e-10, 1e-14]
        nspreads = [compute_kernel_params(tol).nspread for tol in tols]

        # Should be non-decreasing
        for i in range(len(nspreads) - 1):
            assert nspreads[i] <= nspreads[i + 1]

    def test_params_named_tuple(self):
        """Result should be a proper NamedTuple."""
        params = compute_kernel_params(1e-6)

        assert isinstance(params, KernelParams)
        assert hasattr(params, "nspread")
        assert hasattr(params, "beta")
        assert hasattr(params, "c")
        assert hasattr(params, "upsampfac")

    def test_params_custom_upsampfac(self):
        """Should respect custom upsampling factor."""
        params_2 = compute_kernel_params(1e-6, upsampfac=2.0)
        params_1_5 = compute_kernel_params(1e-6, upsampfac=1.5)

        assert params_2.upsampfac == 2.0
        assert params_1_5.upsampfac == 1.5

    def test_params_extreme_tolerances(self):
        """Should handle extreme tolerance values gracefully."""
        # Very loose tolerance
        params_loose = compute_kernel_params(0.5)
        assert params_loose.nspread >= 2

        # Very tight tolerance
        params_tight = compute_kernel_params(1e-15)
        assert params_tight.nspread <= 16


# ============================================================================
# Test Kernel Fourier Series
# ============================================================================


class TestKernelFourierSeries:
    """Tests for kernel_fourier_series function."""

    def test_fourier_series_shape(self):
        """Fourier series should have correct output shape."""
        nf = 64
        nspread = 6
        beta = 10.0
        c = 4.0 / (nspread**2)

        phihat = kernel_fourier_series(nf, nspread, beta, c)
        assert phihat.shape == (nf // 2 + 1,)

    def test_fourier_series_positive(self):
        """Fourier coefficients should be positive (kernel is positive)."""
        nf = 64
        nspread = 6
        beta = 10.0
        c = 4.0 / (nspread**2)

        phihat = kernel_fourier_series(nf, nspread, beta, c)
        # First few coefficients should be positive
        assert jnp.all(phihat[:10] > 0)

    def test_fourier_series_decay(self):
        """Fourier coefficients should decay for large k."""
        nf = 128
        nspread = 8
        beta = 15.0
        c = 4.0 / (nspread**2)

        phihat = kernel_fourier_series(nf, nspread, beta, c)

        # DC component should be largest
        assert phihat[0] >= phihat[1]

        # Should decay (not necessarily monotonically, but overall)
        assert phihat[0] > phihat[-1]

    def test_fourier_series_jit(self):
        """Fourier series computation should be JIT-compilable."""
        # Note: static_argnums in the function definition handles this
        nf = 64
        nspread = 6
        beta = 10.0
        c = 4.0 / (nspread**2)

        phihat1 = kernel_fourier_series(nf, nspread, beta, c)
        phihat2 = kernel_fourier_series(nf, nspread, beta, c)

        assert_allclose(phihat1, phihat2, rtol=1e-14)

    @pytest.mark.parametrize("nf", [32, 64, 128, 256])
    def test_fourier_series_different_sizes(self, nf):
        """Test Fourier series for different grid sizes."""
        nspread = 6
        beta = 10.0
        c = 4.0 / (nspread**2)

        phihat = kernel_fourier_series(nf, nspread, beta, c)

        assert phihat.shape == (nf // 2 + 1,)
        assert jnp.all(jnp.isfinite(phihat))


# ============================================================================
# Test Kernel Integration (Normalization)
# ============================================================================


class TestKernelNormalization:
    """Test kernel normalization and integral properties."""

    def test_kernel_integral(self):
        """Test that kernel integrates to expected value."""
        nspread = 6
        beta = 10.0
        c = 4.0 / (nspread**2)

        # Numerical integration using trapezoidal rule
        z = jnp.linspace(-nspread / 2, nspread / 2, 1000)
        phi = es_kernel(z, beta, c)
        integral = jnp.trapezoid(phi, z)

        # Integral should be positive and finite
        assert integral > 0
        assert jnp.isfinite(integral)

    def test_kernel_mass_vs_beta(self):
        """Larger beta should lead to more concentrated kernel (smaller integral)."""
        nspread = 6
        c = 4.0 / (nspread**2)
        z = jnp.linspace(-nspread / 2, nspread / 2, 1000)

        integrals = []
        for beta in [5.0, 10.0, 15.0, 20.0]:
            phi = es_kernel(z, beta, c)
            integrals.append(float(jnp.trapezoid(phi, z)))

        # Larger beta -> smaller integral (more concentrated)
        for i in range(len(integrals) - 1):
            assert integrals[i] > integrals[i + 1]


# ============================================================================
# Test Kernel Consistency with JAX Transforms
# ============================================================================


class TestKernelJAXTransforms:
    """Test kernel compatibility with JAX transforms."""

    def test_vmap_over_z(self):
        """vmap should work over kernel evaluations."""
        beta = 10.0
        c = 0.25

        @jax.jit
        @jax.vmap
        def batched_kernel(z):
            return es_kernel(z, beta, c)

        z_batch = jnp.linspace(-1, 1, 10).reshape(10, 1)
        result = batched_kernel(z_batch)

        assert result.shape == (10, 1)

    def test_jvp_kernel(self):
        """Forward-mode AD should work on kernel."""

        def kernel_fn(z):
            return es_kernel(z, 10.0, 0.25)

        z = jnp.array([0.5])
        tangent = jnp.array([1.0])

        primals, tangents = jax.jvp(kernel_fn, (z,), (tangent,))

        assert jnp.all(jnp.isfinite(primals))
        assert jnp.all(jnp.isfinite(tangents))

    def test_vjp_kernel(self):
        """Reverse-mode AD should work on kernel."""

        def kernel_fn(z):
            return es_kernel(z, 10.0, 0.25)

        z = jnp.array([0.3, 0.7, 1.0])
        primals, vjp_fn = jax.vjp(kernel_fn, z)
        cotangent = jnp.ones_like(primals)
        (grad,) = vjp_fn(cotangent)

        assert grad.shape == z.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_hessian_kernel(self):
        """Second derivatives should be computable."""

        def scalar_kernel(z):
            return jnp.sum(es_kernel(z, 10.0, 0.25))

        z = jnp.array([0.3, 0.7])
        hess = jax.hessian(scalar_kernel)(z)

        assert hess.shape == (2, 2)
        assert jnp.all(jnp.isfinite(hess))
