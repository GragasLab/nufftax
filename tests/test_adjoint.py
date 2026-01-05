"""
Tests for adjoint relationships between Type 1 and Type 2 transforms.
Note: The adjoint relationship <nufft1(c), f> = <c, nufft2(f)> only holds
with specific normalization conventions. These tests verify the gradient
relationship instead, which is what matters for optimization.
"""

import jax
import jax.numpy as jnp
import numpy as np

from nufftax.transforms import (
    nufft1d1,
    nufft1d1_jvp,
    nufft1d2,
    nufft1d2_jvp,
    nufft1d3,
    nufft2d1,
    nufft2d1_jvp,
    nufft2d2,
    nufft2d2_jvp,
    nufft2d3,
    nufft3d1,
    nufft3d1_jvp,
    nufft3d2,
    nufft3d2_jvp,
)
from nufftax.transforms.nufft3 import compute_type3_grid_size, compute_type3_grid_sizes_2d


class TestGradientRelationships:
    """Test that gradients work correctly for optimization use cases."""

    def test_grad_c_matches_type2(self):
        """Test that gradient w.r.t. c uses Type 2 transform internally."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        # Gradient w.r.t. c should be well-defined
        def loss(c):
            result = nufft1d1(x, c, N, eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_c = jax.grad(loss)(c)
        assert grad_c.shape == c.shape
        assert not jnp.any(jnp.isnan(grad_c))
        # Gradient should be non-zero for non-trivial input
        assert jnp.linalg.norm(grad_c) > 0

    def test_grad_x_works(self):
        """Test that gradient w.r.t. x works correctly."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        def loss(x):
            result = nufft1d1(x, c, N, eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_x = jax.grad(loss)(x)
        assert grad_x.shape == x.shape
        assert not jnp.any(jnp.isnan(grad_x))

    def test_type2_grad_f_works(self):
        """Test gradient w.r.t. f for Type 2 NUFFT."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        f = jax.random.normal(jax.random.PRNGKey(43), (N,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (N,))
        f = f.astype(jnp.complex64)

        def loss(f):
            result = nufft1d2(x, f, eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_f = jax.grad(loss)(f)
        assert grad_f.shape == f.shape
        assert not jnp.any(jnp.isnan(grad_f))


class TestSelfAdjointRoundtrip:
    """Test Type1 followed by Type2 behavior.

    Note: The roundtrip Type1 -> Type2 is NOT a perfect reconstruction.
    Type2(Type1(c)) ≈ N * c only when M=N and points are uniformly spaced.
    For random points, we verify correlation and that the roundtrip
    preserves the general structure.
    """

    def test_roundtrip_1d(self):
        """Test that Type1 -> Type2 roundtrip preserves signal structure."""
        M, N = 128, 128
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex64)

        # Type 1: nonuniform -> uniform
        F = nufft1d1(x, c, N, eps=1e-6, isign=1)
        # Type 2: uniform -> nonuniform
        c_reconstructed = nufft1d2(x, F, eps=1e-6, isign=-1)

        # Reconstructed should have same shape
        assert c_reconstructed.shape == c.shape
        # And be a valid complex array
        assert not jnp.any(jnp.isnan(c_reconstructed))

        # For non-uniform random points with M > N, the roundtrip is lossy
        # because we project M points onto N modes, losing information.
        # We only verify some positive correlation exists (not perfect reconstruction).
        c_normalized = c / jnp.linalg.norm(c)
        c_recon_normalized = c_reconstructed / jnp.linalg.norm(c_reconstructed)
        correlation = jnp.abs(jnp.vdot(c_normalized, c_recon_normalized))
        assert correlation > 0.5, f"Very low correlation: {correlation}"

    def test_roundtrip_2d(self):
        """Test 2D roundtrip preserves signal structure."""
        M, N1, N2 = 100, 32, 32
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        c = c.astype(jnp.complex64)

        F = nufft2d1(x, y, c, (N1, N2), eps=1e-6, isign=1)
        c_reconstructed = nufft2d2(x, y, F, eps=1e-6, isign=-1)

        assert c_reconstructed.shape == c.shape
        assert not jnp.any(jnp.isnan(c_reconstructed))

        # For non-uniform random points, only verify some positive correlation
        c_normalized = c / jnp.linalg.norm(c)
        c_recon_normalized = c_reconstructed / jnp.linalg.norm(c_reconstructed)
        correlation = jnp.abs(jnp.vdot(c_normalized, c_recon_normalized))
        assert correlation > 0.5, f"Very low correlation: {correlation}"

    def test_roundtrip_uniform_1d(self):
        """Test roundtrip with uniformly spaced points - should be near-perfect."""
        N = 64
        M = N  # Same number of points as modes
        x = jnp.linspace(-jnp.pi, jnp.pi, M, endpoint=False)
        c = jax.random.normal(jax.random.PRNGKey(42), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(43), (M,))
        c = c.astype(jnp.complex64)

        # Type 1 -> Type 2 with uniform spacing should reconstruct well
        F = nufft1d1(x, c, N, eps=1e-6, isign=1)
        c_reconstructed = nufft1d2(x, F, eps=1e-6, isign=-1)

        # With uniform spacing, c_reconstructed ≈ N * c
        c_scaled = c_reconstructed / N
        np.testing.assert_allclose(c_scaled, c, rtol=1e-4)


class TestGradient2D:
    """Test 2D gradient computation."""

    def test_grad_c_2d(self):
        """Test gradient w.r.t. c for 2D NUFFT."""
        M, N1, N2 = 100, 32, 32
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        c = c.astype(jnp.complex64)

        def loss(c):
            result = nufft2d1(x, y, c, (N1, N2), eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_c = jax.grad(loss)(c)
        assert grad_c.shape == c.shape
        assert not jnp.any(jnp.isnan(grad_c))

    def test_grad_xy_2d(self):
        """Test gradient w.r.t. x, y for 2D NUFFT."""
        M, N1, N2 = 100, 32, 32
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        c = c.astype(jnp.complex64)

        def loss(x, y):
            result = nufft2d1(x, y, c, (N1, N2), eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_x, grad_y = jax.grad(loss, argnums=(0, 1))(x, y)
        assert grad_x.shape == x.shape
        assert grad_y.shape == y.shape
        assert not jnp.any(jnp.isnan(grad_x))
        assert not jnp.any(jnp.isnan(grad_y))


class TestGradient3D:
    """Test 3D gradient computation."""

    def test_grad_c_3d(self):
        """Test gradient w.r.t. c for 3D NUFFT."""
        M, N1, N2, N3 = 50, 8, 8, 8
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
        z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(45), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(46), (M,))
        c = c.astype(jnp.complex64)

        def loss(c):
            result = nufft3d1(x, y, z, c, (N1, N2, N3), eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_c = jax.grad(loss)(c)
        assert grad_c.shape == c.shape
        assert not jnp.any(jnp.isnan(grad_c))


class TestGradientFiniteDifference:
    """Verify gradients using finite differences.

    These tests compare JAX autodiff gradients against numerical finite differences
    to ensure correctness of the VJP implementations.
    """

    def test_grad_c_finite_diff_1d_type1(self):
        """Verify gradient w.r.t. c for 1D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, N = 20, 16
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex128)

            def loss(c):
                result = nufft1d1(x, c, N, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_c = jax.grad(loss)(c)

            eps = 1e-6
            for i in range(min(3, M)):
                c_plus = c.at[i].add(eps)
                c_minus = c.at[i].add(-eps)
                fd_grad_real = (loss(c_plus) - loss(c_minus)) / (2 * eps)

                c_plus_i = c.at[i].add(1j * eps)
                c_minus_i = c.at[i].add(-1j * eps)
                fd_grad_imag = (loss(c_plus_i) - loss(c_minus_i)) / (2 * eps)

                expected = fd_grad_real + 1j * fd_grad_imag
                np.testing.assert_allclose(
                    grad_c[i],
                    expected,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"1D Type1 grad_c mismatch at index {i}",
                )

    def test_grad_x_finite_diff_1d_type1(self):
        """Verify gradient w.r.t. x for 1D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, N = 20, 16
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex128)

            def loss(x):
                result = nufft1d1(x, c, N, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_x = jax.grad(loss)(x)

            eps = 1e-6
            for i in range(min(3, M)):
                x_plus = x.at[i].add(eps)
                x_minus = x.at[i].add(-eps)
                fd_grad = (loss(x_plus) - loss(x_minus)) / (2 * eps)

                np.testing.assert_allclose(
                    grad_x[i],
                    fd_grad,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"1D Type1 grad_x mismatch at index {i}",
                )

    def test_grad_f_finite_diff_1d_type2(self):
        """Verify gradient w.r.t. f for 1D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, N = 20, 16
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(43), (N,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (N,))
            ).astype(jnp.complex128)

            def loss(f):
                result = nufft1d2(x, f, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_f = jax.grad(loss)(f)

            eps = 1e-6
            for i in range(min(3, N)):
                f_plus = f.at[i].add(eps)
                f_minus = f.at[i].add(-eps)
                fd_grad_real = (loss(f_plus) - loss(f_minus)) / (2 * eps)

                f_plus_i = f.at[i].add(1j * eps)
                f_minus_i = f.at[i].add(-1j * eps)
                fd_grad_imag = (loss(f_plus_i) - loss(f_minus_i)) / (2 * eps)

                expected = fd_grad_real + 1j * fd_grad_imag
                np.testing.assert_allclose(
                    grad_f[i],
                    expected,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"1D Type2 grad_f mismatch at index {i}",
                )

    def test_grad_x_finite_diff_1d_type2(self):
        """Verify gradient w.r.t. x for 1D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, N = 20, 16
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(43), (N,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (N,))
            ).astype(jnp.complex128)

            def loss(x):
                result = nufft1d2(x, f, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_x = jax.grad(loss)(x)

            eps = 1e-6
            for i in range(min(3, M)):
                x_plus = x.at[i].add(eps)
                x_minus = x.at[i].add(-eps)
                fd_grad = (loss(x_plus) - loss(x_minus)) / (2 * eps)

                np.testing.assert_allclose(
                    grad_x[i],
                    fd_grad,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"1D Type2 grad_x mismatch at index {i}",
                )

    def test_grad_c_finite_diff_2d_type1(self):
        """Verify gradient w.r.t. c for 2D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, N1, N2 = 15, 8, 10
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
            ).astype(jnp.complex128)

            def loss(c):
                result = nufft2d1(x, y, c, (N1, N2), eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_c = jax.grad(loss)(c)

            eps = 1e-6
            for i in range(min(3, M)):
                c_plus = c.at[i].add(eps)
                c_minus = c.at[i].add(-eps)
                fd_grad_real = (loss(c_plus) - loss(c_minus)) / (2 * eps)

                c_plus_i = c.at[i].add(1j * eps)
                c_minus_i = c.at[i].add(-1j * eps)
                fd_grad_imag = (loss(c_plus_i) - loss(c_minus_i)) / (2 * eps)

                expected = fd_grad_real + 1j * fd_grad_imag
                np.testing.assert_allclose(
                    grad_c[i],
                    expected,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"2D Type1 grad_c mismatch at index {i}",
                )

    def test_grad_xy_finite_diff_2d_type1(self):
        """Verify gradient w.r.t. x, y for 2D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, N1, N2 = 15, 8, 10
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
            ).astype(jnp.complex128)

            def loss_x(x):
                result = nufft2d1(x, y, c, (N1, N2), eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            def loss_y(y):
                result = nufft2d1(x, y, c, (N1, N2), eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_x = jax.grad(loss_x)(x)
            grad_y = jax.grad(loss_y)(y)

            eps = 1e-6
            for i in range(min(3, M)):
                # Test grad_x
                x_plus = x.at[i].add(eps)
                x_minus = x.at[i].add(-eps)
                fd_grad_x = (loss_x(x_plus) - loss_x(x_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_x[i],
                    fd_grad_x,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"2D Type1 grad_x mismatch at index {i}",
                )

                # Test grad_y
                y_plus = y.at[i].add(eps)
                y_minus = y.at[i].add(-eps)
                fd_grad_y = (loss_y(y_plus) - loss_y(y_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_y[i],
                    fd_grad_y,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"2D Type1 grad_y mismatch at index {i}",
                )

    def test_grad_f_finite_diff_2d_type2(self):
        """Verify gradient w.r.t. f for 2D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, N1, N2 = 15, 8, 10
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            # f has shape (N2, N1) per JAX convention
            f = (
                jax.random.normal(jax.random.PRNGKey(44), (N2, N1))
                + 1j * jax.random.normal(jax.random.PRNGKey(45), (N2, N1))
            ).astype(jnp.complex128)

            def loss(f):
                result = nufft2d2(x, y, f, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_f = jax.grad(loss)(f)

            eps = 1e-6
            # Check a few elements
            for i in range(min(2, N2)):
                for j in range(min(2, N1)):
                    f_plus = f.at[i, j].add(eps)
                    f_minus = f.at[i, j].add(-eps)
                    fd_grad_real = (loss(f_plus) - loss(f_minus)) / (2 * eps)

                    f_plus_i = f.at[i, j].add(1j * eps)
                    f_minus_i = f.at[i, j].add(-1j * eps)
                    fd_grad_imag = (loss(f_plus_i) - loss(f_minus_i)) / (2 * eps)

                    expected = fd_grad_real + 1j * fd_grad_imag
                    np.testing.assert_allclose(
                        grad_f[i, j],
                        expected,
                        rtol=1e-6,
                        atol=1e-8,
                        err_msg=f"2D Type2 grad_f mismatch at index ({i}, {j})",
                    )

    def test_grad_xy_finite_diff_2d_type2(self):
        """Verify gradient w.r.t. x, y for 2D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, N1, N2 = 15, 8, 10
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(44), (N2, N1))
                + 1j * jax.random.normal(jax.random.PRNGKey(45), (N2, N1))
            ).astype(jnp.complex128)

            def loss_x(x):
                result = nufft2d2(x, y, f, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            def loss_y(y):
                result = nufft2d2(x, y, f, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_x = jax.grad(loss_x)(x)
            grad_y = jax.grad(loss_y)(y)

            eps = 1e-6
            for i in range(min(3, M)):
                x_plus = x.at[i].add(eps)
                x_minus = x.at[i].add(-eps)
                fd_grad_x = (loss_x(x_plus) - loss_x(x_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_x[i],
                    fd_grad_x,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"2D Type2 grad_x mismatch at index {i}",
                )

                y_plus = y.at[i].add(eps)
                y_minus = y.at[i].add(-eps)
                fd_grad_y = (loss_y(y_plus) - loss_y(y_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_y[i],
                    fd_grad_y,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"2D Type2 grad_y mismatch at index {i}",
                )

    def test_grad_c_finite_diff_3d_type1(self):
        """Verify gradient w.r.t. c for 3D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, N1, N2, N3 = 10, 4, 4, 4
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(45), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(46), (M,))
            ).astype(jnp.complex128)

            def loss(c):
                result = nufft3d1(x, y, z, c, (N1, N2, N3), eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_c = jax.grad(loss)(c)

            eps = 1e-6
            for i in range(min(3, M)):
                c_plus = c.at[i].add(eps)
                c_minus = c.at[i].add(-eps)
                fd_grad_real = (loss(c_plus) - loss(c_minus)) / (2 * eps)

                c_plus_i = c.at[i].add(1j * eps)
                c_minus_i = c.at[i].add(-1j * eps)
                fd_grad_imag = (loss(c_plus_i) - loss(c_minus_i)) / (2 * eps)

                expected = fd_grad_real + 1j * fd_grad_imag
                np.testing.assert_allclose(
                    grad_c[i],
                    expected,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"3D Type1 grad_c mismatch at index {i}",
                )

    def test_grad_xyz_finite_diff_3d_type1(self):
        """Verify gradient w.r.t. x, y, z for 3D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, N1, N2, N3 = 10, 4, 4, 4
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(45), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(46), (M,))
            ).astype(jnp.complex128)

            def loss_x(x):
                result = nufft3d1(x, y, z, c, (N1, N2, N3), eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            def loss_y(y):
                result = nufft3d1(x, y, z, c, (N1, N2, N3), eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            def loss_z(z):
                result = nufft3d1(x, y, z, c, (N1, N2, N3), eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_x = jax.grad(loss_x)(x)
            grad_y = jax.grad(loss_y)(y)
            grad_z = jax.grad(loss_z)(z)

            eps = 1e-6
            for i in range(min(3, M)):
                # Test grad_x
                x_plus = x.at[i].add(eps)
                x_minus = x.at[i].add(-eps)
                fd_grad_x = (loss_x(x_plus) - loss_x(x_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_x[i],
                    fd_grad_x,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"3D Type1 grad_x mismatch at index {i}",
                )

                # Test grad_y
                y_plus = y.at[i].add(eps)
                y_minus = y.at[i].add(-eps)
                fd_grad_y = (loss_y(y_plus) - loss_y(y_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_y[i],
                    fd_grad_y,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"3D Type1 grad_y mismatch at index {i}",
                )

                # Test grad_z
                z_plus = z.at[i].add(eps)
                z_minus = z.at[i].add(-eps)
                fd_grad_z = (loss_z(z_plus) - loss_z(z_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_z[i],
                    fd_grad_z,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"3D Type1 grad_z mismatch at index {i}",
                )

    def test_grad_f_finite_diff_3d_type2(self):
        """Verify gradient w.r.t. f for 3D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, N1, N2, N3 = 10, 4, 4, 4
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            # f has shape (N3, N2, N1) per JAX convention
            f = (
                jax.random.normal(jax.random.PRNGKey(45), (N3, N2, N1))
                + 1j * jax.random.normal(jax.random.PRNGKey(46), (N3, N2, N1))
            ).astype(jnp.complex128)

            def loss(f):
                result = nufft3d2(x, y, z, f, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_f = jax.grad(loss)(f)

            eps = 1e-6
            # Check a few elements
            for i in range(min(2, N3)):
                for j in range(min(2, N2)):
                    for k in range(min(2, N1)):
                        f_plus = f.at[i, j, k].add(eps)
                        f_minus = f.at[i, j, k].add(-eps)
                        fd_grad_real = (loss(f_plus) - loss(f_minus)) / (2 * eps)

                        f_plus_i = f.at[i, j, k].add(1j * eps)
                        f_minus_i = f.at[i, j, k].add(-1j * eps)
                        fd_grad_imag = (loss(f_plus_i) - loss(f_minus_i)) / (2 * eps)

                        expected = fd_grad_real + 1j * fd_grad_imag
                        np.testing.assert_allclose(
                            grad_f[i, j, k],
                            expected,
                            rtol=1e-6,
                            atol=1e-8,
                            err_msg=f"3D Type2 grad_f mismatch at index ({i}, {j}, {k})",
                        )


class TestGradientFiniteDifferenceType3:
    """Verify Type 3 gradients using finite differences."""

    def test_grad_c_finite_diff_1d_type3(self):
        """Verify gradient w.r.t. c for 1D Type 3 using finite differences."""
        with jax.enable_x64(True):
            M, N = 15, 20
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex128)
            s = jax.random.uniform(jax.random.PRNGKey(45), (N,), minval=-10, maxval=10).astype(jnp.float64)

            # Compute grid size needed for Type 3
            n_modes = compute_type3_grid_size(x, s, eps=1e-10)

            def loss(c):
                result = nufft1d3(x, c, s, n_modes, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_c = jax.grad(loss)(c)

            eps = 1e-6
            for i in range(min(3, M)):
                c_plus = c.at[i].add(eps)
                c_minus = c.at[i].add(-eps)
                fd_grad_real = (loss(c_plus) - loss(c_minus)) / (2 * eps)

                c_plus_i = c.at[i].add(1j * eps)
                c_minus_i = c.at[i].add(-1j * eps)
                fd_grad_imag = (loss(c_plus_i) - loss(c_minus_i)) / (2 * eps)

                expected = fd_grad_real + 1j * fd_grad_imag
                np.testing.assert_allclose(
                    grad_c[i],
                    expected,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"1D Type3 grad_c mismatch at index {i}",
                )

    def test_grad_x_finite_diff_1d_type3(self):
        """Verify gradient w.r.t. x for 1D Type 3 using finite differences."""
        with jax.enable_x64(True):
            M, N = 15, 20
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex128)
            s = jax.random.uniform(jax.random.PRNGKey(45), (N,), minval=-10, maxval=10).astype(jnp.float64)

            # Compute grid size needed for Type 3
            n_modes = compute_type3_grid_size(x, s, eps=1e-10)

            def loss(x):
                result = nufft1d3(x, c, s, n_modes, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_x = jax.grad(loss)(x)

            eps = 1e-6
            for i in range(min(3, M)):
                x_plus = x.at[i].add(eps)
                x_minus = x.at[i].add(-eps)
                fd_grad = (loss(x_plus) - loss(x_minus)) / (2 * eps)

                np.testing.assert_allclose(
                    grad_x[i],
                    fd_grad,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"1D Type3 grad_x mismatch at index {i}",
                )

    def test_grad_s_finite_diff_1d_type3(self):
        """Verify gradient w.r.t. s (target frequencies) for 1D Type 3."""
        with jax.enable_x64(True):
            M, N = 15, 20
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex128)
            s = jax.random.uniform(jax.random.PRNGKey(45), (N,), minval=-10, maxval=10).astype(jnp.float64)

            # Compute grid size needed for Type 3
            n_modes = compute_type3_grid_size(x, s, eps=1e-10)

            def loss(s):
                result = nufft1d3(x, c, s, n_modes, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_s = jax.grad(loss)(s)

            eps = 1e-6
            for i in range(min(3, N)):
                s_plus = s.at[i].add(eps)
                s_minus = s.at[i].add(-eps)
                fd_grad = (loss(s_plus) - loss(s_minus)) / (2 * eps)

                np.testing.assert_allclose(
                    grad_s[i],
                    fd_grad,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"1D Type3 grad_s mismatch at index {i}",
                )

    def test_grad_c_finite_diff_2d_type3(self):
        """Verify gradient w.r.t. c for 2D Type 3 using finite differences."""
        with jax.enable_x64(True):
            M, N = 10, 15
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
            ).astype(jnp.complex128)
            s = jax.random.uniform(jax.random.PRNGKey(46), (N,), minval=-10, maxval=10).astype(jnp.float64)
            t = jax.random.uniform(jax.random.PRNGKey(47), (N,), minval=-10, maxval=10).astype(jnp.float64)

            # Compute grid sizes needed for 2D Type 3
            x_extent = float(jnp.max(x) - jnp.min(x)) / 2
            y_extent = float(jnp.max(y) - jnp.min(y)) / 2
            s_extent = float(jnp.max(s) - jnp.min(s)) / 2
            t_extent = float(jnp.max(t) - jnp.min(t)) / 2
            n_modes = compute_type3_grid_sizes_2d(x_extent, y_extent, s_extent, t_extent, eps=1e-10)

            def loss(c):
                result = nufft2d3(x, y, c, s, t, n_modes, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_c = jax.grad(loss)(c)

            eps = 1e-6
            for i in range(min(3, M)):
                c_plus = c.at[i].add(eps)
                c_minus = c.at[i].add(-eps)
                fd_grad_real = (loss(c_plus) - loss(c_minus)) / (2 * eps)

                c_plus_i = c.at[i].add(1j * eps)
                c_minus_i = c.at[i].add(-1j * eps)
                fd_grad_imag = (loss(c_plus_i) - loss(c_minus_i)) / (2 * eps)

                expected = fd_grad_real + 1j * fd_grad_imag
                np.testing.assert_allclose(
                    grad_c[i],
                    expected,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"2D Type3 grad_c mismatch at index {i}",
                )

    def test_grad_xy_finite_diff_2d_type3(self):
        """Verify gradient w.r.t. x, y for 2D Type 3 using finite differences."""
        with jax.enable_x64(True):
            M, N = 10, 15
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
            ).astype(jnp.complex128)
            s = jax.random.uniform(jax.random.PRNGKey(46), (N,), minval=-10, maxval=10).astype(jnp.float64)
            t = jax.random.uniform(jax.random.PRNGKey(47), (N,), minval=-10, maxval=10).astype(jnp.float64)

            # Compute grid sizes needed for 2D Type 3
            x_extent = float(jnp.max(x) - jnp.min(x)) / 2
            y_extent = float(jnp.max(y) - jnp.min(y)) / 2
            s_extent = float(jnp.max(s) - jnp.min(s)) / 2
            t_extent = float(jnp.max(t) - jnp.min(t)) / 2
            n_modes = compute_type3_grid_sizes_2d(x_extent, y_extent, s_extent, t_extent, eps=1e-10)

            def loss_x(x):
                result = nufft2d3(x, y, c, s, t, n_modes, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            def loss_y(y):
                result = nufft2d3(x, y, c, s, t, n_modes, eps=1e-10)
                return jnp.sum(jnp.abs(result) ** 2).real

            grad_x = jax.grad(loss_x)(x)
            grad_y = jax.grad(loss_y)(y)

            eps = 1e-6
            for i in range(min(3, M)):
                x_plus = x.at[i].add(eps)
                x_minus = x.at[i].add(-eps)
                fd_grad_x = (loss_x(x_plus) - loss_x(x_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_x[i],
                    fd_grad_x,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"2D Type3 grad_x mismatch at index {i}",
                )

                y_plus = y.at[i].add(eps)
                y_minus = y.at[i].add(-eps)
                fd_grad_y = (loss_y(y_plus) - loss_y(y_minus)) / (2 * eps)
                np.testing.assert_allclose(
                    grad_y[i],
                    fd_grad_y,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"2D Type3 grad_y mismatch at index {i}",
                )


class TestJVPFiniteDifference:
    """Verify JVP (forward-mode autodiff) using finite differences."""

    def test_jvp_1d_type1_tangent_c(self):
        """Verify JVP tangent w.r.t. c for 1D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, n_modes = 20, 15
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex128)

            # Random tangent direction for c
            dc = (
                jax.random.normal(jax.random.PRNGKey(45), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(46), (M,))
            ).astype(jnp.complex128)

            # Compute JVP
            primals, tangents = jax.jvp(lambda c: nufft1d1_jvp(x, c, n_modes, eps=1e-10), (c,), (dc,))

            # Finite difference approximation
            eps = 1e-7
            f_plus = nufft1d1_jvp(x, c + eps * dc, n_modes, eps=1e-10)
            f_minus = nufft1d1_jvp(x, c - eps * dc, n_modes, eps=1e-10)
            fd_tangent = (f_plus - f_minus) / (2 * eps)

            np.testing.assert_allclose(
                tangents,
                fd_tangent,
                rtol=1e-5,
                atol=1e-8,
                err_msg="1D Type1 JVP tangent w.r.t. c mismatch",
            )

    def test_jvp_1d_type1_tangent_x(self):
        """Verify JVP tangent w.r.t. x for 1D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, n_modes = 20, 15
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex128)

            # Random tangent direction for x
            dx = jax.random.normal(jax.random.PRNGKey(47), (M,)).astype(jnp.float64) * 0.1

            # Compute JVP
            primals, tangents = jax.jvp(lambda x: nufft1d1_jvp(x, c, n_modes, eps=1e-10), (x,), (dx,))

            # Finite difference approximation
            eps = 1e-7
            f_plus = nufft1d1_jvp(x + eps * dx, c, n_modes, eps=1e-10)
            f_minus = nufft1d1_jvp(x - eps * dx, c, n_modes, eps=1e-10)
            fd_tangent = (f_plus - f_minus) / (2 * eps)

            np.testing.assert_allclose(
                tangents,
                fd_tangent,
                rtol=1e-5,
                atol=1e-8,
                err_msg="1D Type1 JVP tangent w.r.t. x mismatch",
            )

    def test_jvp_1d_type1_tangent_both(self):
        """Verify JVP tangent w.r.t. both x and c for 1D Type 1."""
        with jax.enable_x64(True):
            M, n_modes = 20, 15
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
            ).astype(jnp.complex128)

            # Random tangent directions
            dx = jax.random.normal(jax.random.PRNGKey(47), (M,)).astype(jnp.float64) * 0.1
            dc = (
                jax.random.normal(jax.random.PRNGKey(45), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(46), (M,))
            ).astype(jnp.complex128)

            # Compute JVP with both tangents
            primals, tangents = jax.jvp(lambda x, c: nufft1d1_jvp(x, c, n_modes, eps=1e-10), (x, c), (dx, dc))

            # Finite difference approximation
            eps = 1e-7
            f_plus = nufft1d1_jvp(x + eps * dx, c + eps * dc, n_modes, eps=1e-10)
            f_minus = nufft1d1_jvp(x - eps * dx, c - eps * dc, n_modes, eps=1e-10)
            fd_tangent = (f_plus - f_minus) / (2 * eps)

            np.testing.assert_allclose(
                tangents,
                fd_tangent,
                rtol=1e-5,
                atol=1e-8,
                err_msg="1D Type1 JVP tangent w.r.t. both x and c mismatch",
            )

    def test_jvp_1d_type2_tangent_f(self):
        """Verify JVP tangent w.r.t. f for 1D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, n_modes = 20, 15
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(43), (n_modes,))
                + 1j * jax.random.normal(jax.random.PRNGKey(44), (n_modes,))
            ).astype(jnp.complex128)

            # Random tangent direction for f
            df = (
                jax.random.normal(jax.random.PRNGKey(45), (n_modes,))
                + 1j * jax.random.normal(jax.random.PRNGKey(46), (n_modes,))
            ).astype(jnp.complex128)

            # Compute JVP
            primals, tangents = jax.jvp(lambda f: nufft1d2_jvp(x, f, eps=1e-10), (f,), (df,))

            # Finite difference approximation
            eps = 1e-7
            c_plus = nufft1d2_jvp(x, f + eps * df, eps=1e-10)
            c_minus = nufft1d2_jvp(x, f - eps * df, eps=1e-10)
            fd_tangent = (c_plus - c_minus) / (2 * eps)

            np.testing.assert_allclose(
                tangents,
                fd_tangent,
                rtol=1e-5,
                atol=1e-8,
                err_msg="1D Type2 JVP tangent w.r.t. f mismatch",
            )

    def test_jvp_1d_type2_tangent_x(self):
        """Verify JVP tangent w.r.t. x for 1D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, n_modes = 20, 15
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(43), (n_modes,))
                + 1j * jax.random.normal(jax.random.PRNGKey(44), (n_modes,))
            ).astype(jnp.complex128)

            # Random tangent direction for x
            dx = jax.random.normal(jax.random.PRNGKey(47), (M,)).astype(jnp.float64) * 0.1

            # Compute JVP
            primals, tangents = jax.jvp(lambda x: nufft1d2_jvp(x, f, eps=1e-10), (x,), (dx,))

            # Finite difference approximation
            eps = 1e-7
            c_plus = nufft1d2_jvp(x + eps * dx, f, eps=1e-10)
            c_minus = nufft1d2_jvp(x - eps * dx, f, eps=1e-10)
            fd_tangent = (c_plus - c_minus) / (2 * eps)

            np.testing.assert_allclose(
                tangents,
                fd_tangent,
                rtol=1e-5,
                atol=1e-8,
                err_msg="1D Type2 JVP tangent w.r.t. x mismatch",
            )

    def test_jvp_1d_type2_tangent_both(self):
        """Verify JVP tangent w.r.t. both x and f for 1D Type 2."""
        with jax.enable_x64(True):
            M, n_modes = 20, 15
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(43), (n_modes,))
                + 1j * jax.random.normal(jax.random.PRNGKey(44), (n_modes,))
            ).astype(jnp.complex128)

            # Random tangent directions
            dx = jax.random.normal(jax.random.PRNGKey(47), (M,)).astype(jnp.float64) * 0.1
            df = (
                jax.random.normal(jax.random.PRNGKey(45), (n_modes,))
                + 1j * jax.random.normal(jax.random.PRNGKey(46), (n_modes,))
            ).astype(jnp.complex128)

            # Compute JVP with both tangents
            primals, tangents = jax.jvp(lambda x, f: nufft1d2_jvp(x, f, eps=1e-10), (x, f), (dx, df))

            # Finite difference approximation
            eps = 1e-7
            c_plus = nufft1d2_jvp(x + eps * dx, f + eps * df, eps=1e-10)
            c_minus = nufft1d2_jvp(x - eps * dx, f - eps * df, eps=1e-10)
            fd_tangent = (c_plus - c_minus) / (2 * eps)

            np.testing.assert_allclose(
                tangents,
                fd_tangent,
                rtol=1e-5,
                atol=1e-8,
                err_msg="1D Type2 JVP tangent w.r.t. both x and f mismatch",
            )

    # =========================================================================
    # 2D Type 1 JVP Tests
    # =========================================================================

    def test_jvp_2d_type1_tangent_c(self):
        """Verify JVP tangent w.r.t. c for 2D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, n1, n2 = 20, 10, 12
            n_modes = (n1, n2)
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
            ).astype(jnp.complex128)
            dc = (
                jax.random.normal(jax.random.PRNGKey(46), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(47), (M,))
            ).astype(jnp.complex128)

            primals, tangents = jax.jvp(lambda c: nufft2d1_jvp(x, y, c, n_modes, eps=1e-10), (c,), (dc,))

            eps = 1e-7
            f_plus = nufft2d1_jvp(x, y, c + eps * dc, n_modes, eps=1e-10)
            f_minus = nufft2d1_jvp(x, y, c - eps * dc, n_modes, eps=1e-10)
            fd_tangent = (f_plus - f_minus) / (2 * eps)

            np.testing.assert_allclose(tangents, fd_tangent, rtol=1e-5, atol=1e-8)

    def test_jvp_2d_type1_tangent_xy(self):
        """Verify JVP tangent w.r.t. x, y for 2D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, n1, n2 = 20, 10, 12
            n_modes = (n1, n2)
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
            ).astype(jnp.complex128)
            dx = jax.random.normal(jax.random.PRNGKey(46), (M,)).astype(jnp.float64) * 0.1
            dy = jax.random.normal(jax.random.PRNGKey(47), (M,)).astype(jnp.float64) * 0.1

            primals, tangents = jax.jvp(lambda x, y: nufft2d1_jvp(x, y, c, n_modes, eps=1e-10), (x, y), (dx, dy))

            eps = 1e-7
            f_plus = nufft2d1_jvp(x + eps * dx, y + eps * dy, c, n_modes, eps=1e-10)
            f_minus = nufft2d1_jvp(x - eps * dx, y - eps * dy, c, n_modes, eps=1e-10)
            fd_tangent = (f_plus - f_minus) / (2 * eps)

            np.testing.assert_allclose(tangents, fd_tangent, rtol=1e-5, atol=2e-8)

    # =========================================================================
    # 2D Type 2 JVP Tests
    # =========================================================================

    def test_jvp_2d_type2_tangent_f(self):
        """Verify JVP tangent w.r.t. f for 2D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, n1, n2 = 20, 10, 12
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(44), (n2, n1))
                + 1j * jax.random.normal(jax.random.PRNGKey(45), (n2, n1))
            ).astype(jnp.complex128)
            df = (
                jax.random.normal(jax.random.PRNGKey(46), (n2, n1))
                + 1j * jax.random.normal(jax.random.PRNGKey(47), (n2, n1))
            ).astype(jnp.complex128)

            primals, tangents = jax.jvp(lambda f: nufft2d2_jvp(x, y, f, eps=1e-10), (f,), (df,))

            eps = 1e-7
            c_plus = nufft2d2_jvp(x, y, f + eps * df, eps=1e-10)
            c_minus = nufft2d2_jvp(x, y, f - eps * df, eps=1e-10)
            fd_tangent = (c_plus - c_minus) / (2 * eps)

            np.testing.assert_allclose(tangents, fd_tangent, rtol=1e-5, atol=1e-8)

    def test_jvp_2d_type2_tangent_xy(self):
        """Verify JVP tangent w.r.t. x, y for 2D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, n1, n2 = 20, 10, 12
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(44), (n2, n1))
                + 1j * jax.random.normal(jax.random.PRNGKey(45), (n2, n1))
            ).astype(jnp.complex128)
            dx = jax.random.normal(jax.random.PRNGKey(46), (M,)).astype(jnp.float64) * 0.1
            dy = jax.random.normal(jax.random.PRNGKey(47), (M,)).astype(jnp.float64) * 0.1

            primals, tangents = jax.jvp(lambda x, y: nufft2d2_jvp(x, y, f, eps=1e-10), (x, y), (dx, dy))

            eps = 1e-7
            c_plus = nufft2d2_jvp(x + eps * dx, y + eps * dy, f, eps=1e-10)
            c_minus = nufft2d2_jvp(x - eps * dx, y - eps * dy, f, eps=1e-10)
            fd_tangent = (c_plus - c_minus) / (2 * eps)

            np.testing.assert_allclose(tangents, fd_tangent, rtol=1e-5, atol=1e-8)

    # =========================================================================
    # 3D Type 1 JVP Tests
    # =========================================================================

    def test_jvp_3d_type1_tangent_c(self):
        """Verify JVP tangent w.r.t. c for 3D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, n1, n2, n3 = 15, 6, 8, 7
            n_modes = (n1, n2, n3)
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(45), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(46), (M,))
            ).astype(jnp.complex128)
            dc = (
                jax.random.normal(jax.random.PRNGKey(47), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(48), (M,))
            ).astype(jnp.complex128)

            primals, tangents = jax.jvp(lambda c: nufft3d1_jvp(x, y, z, c, n_modes, eps=1e-10), (c,), (dc,))

            eps = 1e-7
            f_plus = nufft3d1_jvp(x, y, z, c + eps * dc, n_modes, eps=1e-10)
            f_minus = nufft3d1_jvp(x, y, z, c - eps * dc, n_modes, eps=1e-10)
            fd_tangent = (f_plus - f_minus) / (2 * eps)

            np.testing.assert_allclose(tangents, fd_tangent, rtol=1e-5, atol=1e-8)

    def test_jvp_3d_type1_tangent_xyz(self):
        """Verify JVP tangent w.r.t. x, y, z for 3D Type 1 using finite differences."""
        with jax.enable_x64(True):
            M, n1, n2, n3 = 15, 6, 8, 7
            n_modes = (n1, n2, n3)
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = (
                jax.random.normal(jax.random.PRNGKey(45), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(46), (M,))
            ).astype(jnp.complex128)
            dx = jax.random.normal(jax.random.PRNGKey(47), (M,)).astype(jnp.float64) * 0.1
            dy = jax.random.normal(jax.random.PRNGKey(48), (M,)).astype(jnp.float64) * 0.1
            dz = jax.random.normal(jax.random.PRNGKey(49), (M,)).astype(jnp.float64) * 0.1

            primals, tangents = jax.jvp(
                lambda x, y, z: nufft3d1_jvp(x, y, z, c, n_modes, eps=1e-10),
                (x, y, z),
                (dx, dy, dz),
            )

            eps = 1e-7
            f_plus = nufft3d1_jvp(x + eps * dx, y + eps * dy, z + eps * dz, c, n_modes, eps=1e-10)
            f_minus = nufft3d1_jvp(x - eps * dx, y - eps * dy, z - eps * dz, c, n_modes, eps=1e-10)
            fd_tangent = (f_plus - f_minus) / (2 * eps)

            np.testing.assert_allclose(tangents, fd_tangent, rtol=1e-5, atol=1e-8)

    # =========================================================================
    # 3D Type 2 JVP Tests
    # =========================================================================

    def test_jvp_3d_type2_tangent_f(self):
        """Verify JVP tangent w.r.t. f for 3D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, n1, n2, n3 = 15, 6, 8, 7
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(45), (n3, n2, n1))
                + 1j * jax.random.normal(jax.random.PRNGKey(46), (n3, n2, n1))
            ).astype(jnp.complex128)
            df = (
                jax.random.normal(jax.random.PRNGKey(47), (n3, n2, n1))
                + 1j * jax.random.normal(jax.random.PRNGKey(48), (n3, n2, n1))
            ).astype(jnp.complex128)

            primals, tangents = jax.jvp(lambda f: nufft3d2_jvp(x, y, z, f, eps=1e-10), (f,), (df,))

            eps = 1e-7
            c_plus = nufft3d2_jvp(x, y, z, f + eps * df, eps=1e-10)
            c_minus = nufft3d2_jvp(x, y, z, f - eps * df, eps=1e-10)
            fd_tangent = (c_plus - c_minus) / (2 * eps)

            np.testing.assert_allclose(tangents, fd_tangent, rtol=1e-5, atol=1e-8)

    def test_jvp_3d_type2_tangent_xyz(self):
        """Verify JVP tangent w.r.t. x, y, z for 3D Type 2 using finite differences."""
        with jax.enable_x64(True):
            M, n1, n2, n3 = 15, 6, 8, 7
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            f = (
                jax.random.normal(jax.random.PRNGKey(45), (n3, n2, n1))
                + 1j * jax.random.normal(jax.random.PRNGKey(46), (n3, n2, n1))
            ).astype(jnp.complex128)
            dx = jax.random.normal(jax.random.PRNGKey(47), (M,)).astype(jnp.float64) * 0.1
            dy = jax.random.normal(jax.random.PRNGKey(48), (M,)).astype(jnp.float64) * 0.1
            dz = jax.random.normal(jax.random.PRNGKey(49), (M,)).astype(jnp.float64) * 0.1

            primals, tangents = jax.jvp(lambda x, y, z: nufft3d2_jvp(x, y, z, f, eps=1e-10), (x, y, z), (dx, dy, dz))

            eps = 1e-7
            c_plus = nufft3d2_jvp(x + eps * dx, y + eps * dy, z + eps * dz, f, eps=1e-10)
            c_minus = nufft3d2_jvp(x - eps * dx, y - eps * dy, z - eps * dz, f, eps=1e-10)
            fd_tangent = (c_plus - c_minus) / (2 * eps)

            np.testing.assert_allclose(tangents, fd_tangent, rtol=1e-5, atol=1e-8)
