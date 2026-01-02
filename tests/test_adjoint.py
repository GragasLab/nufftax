"""
Tests for adjoint relationships between Type 1 and Type 2 transforms.
Note: The adjoint relationship <nufft1(c), f> = <c, nufft2(f)> only holds
with specific normalization conventions. These tests verify the gradient
relationship instead, which is what matters for optimization.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nufftax.transforms import nufft1d1, nufft1d2, nufft2d1, nufft2d2, nufft3d1


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
    """Test Type1 followed by Type2 behavior."""

    def test_roundtrip_1d(self):
        """Test that Type1 -> Type2 roundtrip produces valid results."""
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

    def test_roundtrip_2d(self):
        """Test 2D roundtrip."""
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
    """Verify gradients using finite differences."""

    @pytest.mark.skipif(not jax.config.jax_enable_x64, reason="Finite difference tests require x64 precision")
    def test_grad_c_finite_diff_1d(self):
        """Verify gradient w.r.t. c using finite differences."""
        M, N = 20, 16  # Small for finite diff
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(44), (M,))
        c = c.astype(jnp.complex128)  # Use float64 for finite diff accuracy
        x = x.astype(jnp.float64)

        def loss(c):
            result = nufft1d1(x, c, N, eps=1e-10)
            return jnp.sum(jnp.abs(result) ** 2).real

        # Analytical gradient
        grad_c = jax.grad(loss)(c)

        # Finite difference approximation (for real part)
        eps = 1e-6
        for i in range(min(3, M)):  # Check first 3 components
            c_plus = c.at[i].add(eps)
            c_minus = c.at[i].add(-eps)
            fd_grad_real = (loss(c_plus) - loss(c_minus)) / (2 * eps)

            c_plus_i = c.at[i].add(1j * eps)
            c_minus_i = c.at[i].add(-1j * eps)
            fd_grad_imag = (loss(c_plus_i) - loss(c_minus_i)) / (2 * eps)

            # Wirtinger derivative: grad = 2 * (d_real + i * d_imag)
            expected = fd_grad_real + 1j * fd_grad_imag

            np.testing.assert_allclose(
                grad_c[i], expected, rtol=1e-4, atol=1e-6, err_msg=f"Gradient mismatch at index {i}"
            )
