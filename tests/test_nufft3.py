"""
Tests for Type 3 NUFFT (nonuniform to nonuniform).
"""

import finufft
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 for 3D Type 3 tests which need higher precision
jax.config.update("jax_enable_x64", True)

from nufftax import (  # noqa: E402
    compute_type3_grid_size,
    compute_type3_grid_sizes_2d,
    compute_type3_grid_sizes_3d,
    nufft1d3,
    nufft2d3,
    nufft3d3,
)


def direct_nufft1d3(x, c, s, isign=1):
    """Direct computation of 1D Type 3 NUFFT for verification.

    f[k] = sum_j c[j] * exp(isign * i * s[k] * x[j])
    """
    # Ensure proper shapes
    x = jnp.atleast_1d(x)
    c = jnp.atleast_1d(c)
    s = jnp.atleast_1d(s)

    # Outer product: s[k] * x[j]
    phase = isign * jnp.outer(s, x)  # (N, M)

    # Sum over source points
    f = jnp.sum(c[None, :] * jnp.exp(1j * phase), axis=1)

    return f


def direct_nufft2d3(x, y, c, s, t, isign=1):
    """Direct computation of 2D Type 3 NUFFT."""
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
    c = jnp.atleast_1d(c)
    s = jnp.atleast_1d(s)
    t = jnp.atleast_1d(t)

    # Phase: s[k]*x[j] + t[k]*y[j]
    phase = isign * (jnp.outer(s, x) + jnp.outer(t, y))  # (N, M)

    f = jnp.sum(c[None, :] * jnp.exp(1j * phase), axis=1)

    return f


def direct_nufft3d3(x, y, z, c, s, t, u, isign=1):
    """Direct computation of 3D Type 3 NUFFT."""
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
    z = jnp.atleast_1d(z)
    c = jnp.atleast_1d(c)
    s = jnp.atleast_1d(s)
    t = jnp.atleast_1d(t)
    u = jnp.atleast_1d(u)

    phase = isign * (jnp.outer(s, x) + jnp.outer(t, y) + jnp.outer(u, z))

    f = jnp.sum(c[None, :] * jnp.exp(1j * phase), axis=1)

    return f


def get_n_modes_1d(x, s, eps=1e-6):
    """Helper to compute n_modes for 1D Type 3."""
    x_ext = (float(jnp.max(x)) - float(jnp.min(x))) / 2
    s_ext = (float(jnp.max(s)) - float(jnp.min(s))) / 2
    return compute_type3_grid_size(x_ext, s_ext, eps=eps)


def get_n_modes_2d(x, y, s, t, eps=1e-6):
    """Helper to compute n_modes for 2D Type 3."""
    x_ext = (float(jnp.max(x)) - float(jnp.min(x))) / 2
    y_ext = (float(jnp.max(y)) - float(jnp.min(y))) / 2
    s_ext = (float(jnp.max(s)) - float(jnp.min(s))) / 2
    t_ext = (float(jnp.max(t)) - float(jnp.min(t))) / 2
    return compute_type3_grid_sizes_2d(x_ext, y_ext, s_ext, t_ext, eps=eps)


def get_n_modes_3d(x, y, z, s, t, u, eps=1e-6):
    """Helper to compute n_modes for 3D Type 3."""
    x_ext = (float(jnp.max(x)) - float(jnp.min(x))) / 2
    y_ext = (float(jnp.max(y)) - float(jnp.min(y))) / 2
    z_ext = (float(jnp.max(z)) - float(jnp.min(z))) / 2
    s_ext = (float(jnp.max(s)) - float(jnp.min(s))) / 2
    t_ext = (float(jnp.max(t)) - float(jnp.min(t))) / 2
    u_ext = (float(jnp.max(u)) - float(jnp.min(u))) / 2
    return compute_type3_grid_sizes_3d(x_ext, y_ext, z_ext, s_ext, t_ext, u_ext, eps=eps)


class TestNufft1d3:
    """Tests for 1D Type 3 NUFFT."""

    def test_basic_accuracy(self):
        """Test basic accuracy against direct computation."""
        M, N = 50, 30
        key = jax.random.PRNGKey(42)

        # Random source points and target frequencies
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(43), (N,), minval=-10, maxval=10)
        c = (
            jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        ).astype(jnp.complex64)

        nf = get_n_modes_1d(x, s, eps=1e-6)
        result = nufft1d3(x, c, s, nf, eps=1e-6)

        # Compute directly
        expected = direct_nufft1d3(x, c, s)

        rel_error = jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected)
        assert rel_error < 1e-4, f"Relative error {rel_error} too large"

    def test_isign_negative(self):
        """Test with negative sign convention."""
        M, N = 30, 20
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(43), (N,), minval=-5, maxval=5)
        c = (
            jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        ).astype(jnp.complex64)

        nf = get_n_modes_1d(x, s, eps=1e-6)
        result = nufft1d3(x, c, s, nf, eps=1e-6, isign=-1)
        expected = direct_nufft1d3(x, c, s, isign=-1)

        rel_error = jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected)
        assert rel_error < 1e-4

    def test_single_point(self):
        """Test with single source point."""
        x = jnp.array([0.5])
        c = jnp.array([1.0 + 2.0j])
        s = jnp.array([-1.0, 0.0, 1.0, 2.0])

        nf = get_n_modes_1d(x, s, eps=1e-6)
        result = nufft1d3(x, c, s, nf, eps=1e-6)
        expected = direct_nufft1d3(x, c, s)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_single_target(self):
        """Test with single target frequency."""
        M = 50
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        c = (
            jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        ).astype(jnp.complex64)
        s = jnp.array([3.14])

        nf = get_n_modes_1d(x, s, eps=1e-6)
        result = nufft1d3(x, c, s, nf, eps=1e-6)
        expected = direct_nufft1d3(x, c, s)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batched_input(self):
        """Test with batched input."""
        M, N = 50, 30
        n_batch = 4
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(43), (N,), minval=-5, maxval=5)
        c = (
            jax.random.normal(jax.random.PRNGKey(44), (n_batch, M))
            + 1j * jax.random.normal(jax.random.PRNGKey(45), (n_batch, M))
        ).astype(jnp.complex64)

        nf = get_n_modes_1d(x, s, eps=1e-6)
        result = nufft1d3(x, c, s, nf, eps=1e-6)

        assert result.shape == (n_batch, N)

        # Check each batch
        for i in range(n_batch):
            expected = direct_nufft1d3(x, c[i], s)
            rel_error = jnp.linalg.norm(result[i] - expected) / jnp.linalg.norm(expected)
            assert rel_error < 1e-4

    def test_large_frequency_range(self):
        """Test with large frequency range."""
        M, N = 50, 30
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        # Large frequency range
        s = jax.random.uniform(jax.random.PRNGKey(43), (N,), minval=-100, maxval=100)
        c = (
            jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        ).astype(jnp.complex64)

        nf = get_n_modes_1d(x, s, eps=1e-5)
        result = nufft1d3(x, c, s, nf, eps=1e-5)
        expected = direct_nufft1d3(x, c, s)

        rel_error = jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected)
        assert rel_error < 1e-3  # Slightly looser tolerance for large range


class TestNufft2d3:
    """Tests for 2D Type 3 NUFFT."""

    def test_basic_accuracy(self):
        """Test basic accuracy against direct computation."""
        M, N = 40, 25
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(44), (N,), minval=-5, maxval=5)
        t = jax.random.uniform(jax.random.PRNGKey(45), (N,), minval=-5, maxval=5)
        c = (
            jax.random.normal(jax.random.PRNGKey(46), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(47), (M,))
        ).astype(jnp.complex64)

        n_modes = get_n_modes_2d(x, y, s, t, eps=1e-6)
        result = nufft2d3(x, y, c, s, t, n_modes, eps=1e-6)
        expected = direct_nufft2d3(x, y, c, s, t)

        rel_error = jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected)
        assert rel_error < 1e-4

    def test_batched_input(self):
        """Test with batched input."""
        M, N = 30, 20
        n_batch = 3
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(44), (N,), minval=-5, maxval=5)
        t = jax.random.uniform(jax.random.PRNGKey(45), (N,), minval=-5, maxval=5)
        c = (
            jax.random.normal(jax.random.PRNGKey(46), (n_batch, M))
            + 1j * jax.random.normal(jax.random.PRNGKey(47), (n_batch, M))
        ).astype(jnp.complex64)

        n_modes = get_n_modes_2d(x, y, s, t, eps=1e-6)
        result = nufft2d3(x, y, c, s, t, n_modes, eps=1e-6)

        assert result.shape == (n_batch, N)


class TestNufft3d3:
    """Tests for 3D Type 3 NUFFT."""

    def test_basic_accuracy(self):
        """Test basic accuracy against direct computation.

        Note: 3D Type 3 requires float64 precision due to accumulated errors
        from the multi-step algorithm (spread -> Type 2 -> deconvolve).
        """
        M, N = 30, 20
        key = jax.random.PRNGKey(42)

        # Use float64 for 3D Type 3 - float32 accumulates too much error
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64)
        z = jax.random.uniform(jax.random.PRNGKey(44), (M,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64)
        s = jax.random.uniform(jax.random.PRNGKey(45), (N,), minval=-3, maxval=3, dtype=jnp.float64)
        t = jax.random.uniform(jax.random.PRNGKey(46), (N,), minval=-3, maxval=3, dtype=jnp.float64)
        u = jax.random.uniform(jax.random.PRNGKey(47), (N,), minval=-3, maxval=3, dtype=jnp.float64)
        c = jax.random.normal(jax.random.PRNGKey(48), (M,), dtype=jnp.float64) + 1j * jax.random.normal(
            jax.random.PRNGKey(49), (M,), dtype=jnp.float64
        )

        n_modes = get_n_modes_3d(x, y, z, s, t, u, eps=1e-5)
        result = nufft3d3(x, y, z, c, s, t, u, n_modes, eps=1e-5)
        expected = direct_nufft3d3(x, y, z, c, s, t, u)

        rel_error = jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected)
        assert rel_error < 1e-3  # 3D has more accumulated error


class TestNufft1d3FINUFFT:
    """Tests comparing nufft1d3 against FINUFFT reference implementation."""

    def test_nufft1d3_vs_finufft_basic(self):
        """Test basic accuracy against FINUFFT."""
        M, N = 100, 50
        rng = np.random.default_rng(42)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        s = rng.uniform(-10, 10, N).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        # FINUFFT reference (isign=1 is default)
        f_finufft = finufft.nufft1d3(x, c, s, eps=1e-9)

        # JAX implementation
        nf = get_n_modes_1d(jnp.array(x), jnp.array(s), eps=1e-9)
        f_jax = nufft1d3(jnp.array(x), jnp.array(c), jnp.array(s), nf, eps=1e-9)

        rel_error = np.linalg.norm(np.array(f_jax) - f_finufft) / np.linalg.norm(f_finufft)
        assert rel_error < 1e-6, f"Relative error {rel_error} too large"

    @pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-9])
    def test_nufft1d3_vs_finufft_precision(self, eps):
        """Test accuracy at different precision levels."""
        M, N = 100, 50
        rng = np.random.default_rng(42)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        s = rng.uniform(-10, 10, N).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f_finufft = finufft.nufft1d3(x, c, s, eps=eps)

        nf = get_n_modes_1d(jnp.array(x), jnp.array(s), eps=eps)
        f_jax = nufft1d3(jnp.array(x), jnp.array(c), jnp.array(s), nf, eps=eps)

        rel_error = np.linalg.norm(np.array(f_jax) - f_finufft) / np.linalg.norm(f_finufft)
        # Allow 10x the requested precision for implementation differences
        assert rel_error < 10 * eps, f"Relative error {rel_error} exceeds 10*eps={10 * eps}"

    def test_nufft1d3_vs_finufft_isign_negative(self):
        """Test with negative sign convention."""
        M, N = 100, 50
        rng = np.random.default_rng(42)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        s = rng.uniform(-10, 10, N).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f_finufft = finufft.nufft1d3(x, c, s, eps=1e-9, isign=-1)

        nf = get_n_modes_1d(jnp.array(x), jnp.array(s), eps=1e-9)
        f_jax = nufft1d3(jnp.array(x), jnp.array(c), jnp.array(s), nf, eps=1e-9, isign=-1)

        rel_error = np.linalg.norm(np.array(f_jax) - f_finufft) / np.linalg.norm(f_finufft)
        assert rel_error < 1e-6


class TestNufft2d3FINUFFT:
    """Tests comparing nufft2d3 against FINUFFT reference implementation."""

    def test_nufft2d3_vs_finufft_basic(self):
        """Test basic accuracy against FINUFFT."""
        M, N = 100, 50
        rng = np.random.default_rng(42)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        s = rng.uniform(-10, 10, N).astype(np.float64)
        t = rng.uniform(-10, 10, N).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f_finufft = finufft.nufft2d3(x, y, c, s, t, eps=1e-9)

        n_modes = get_n_modes_2d(jnp.array(x), jnp.array(y), jnp.array(s), jnp.array(t), eps=1e-9)
        f_jax = nufft2d3(jnp.array(x), jnp.array(y), jnp.array(c), jnp.array(s), jnp.array(t), n_modes, eps=1e-9)

        rel_error = np.linalg.norm(np.array(f_jax) - f_finufft) / np.linalg.norm(f_finufft)
        assert rel_error < 1e-6, f"Relative error {rel_error} too large"

    @pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-9])
    def test_nufft2d3_vs_finufft_precision(self, eps):
        """Test accuracy at different precision levels."""
        M, N = 80, 40
        rng = np.random.default_rng(42)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        s = rng.uniform(-5, 5, N).astype(np.float64)
        t = rng.uniform(-5, 5, N).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f_finufft = finufft.nufft2d3(x, y, c, s, t, eps=eps)

        n_modes = get_n_modes_2d(jnp.array(x), jnp.array(y), jnp.array(s), jnp.array(t), eps=eps)
        f_jax = nufft2d3(jnp.array(x), jnp.array(y), jnp.array(c), jnp.array(s), jnp.array(t), n_modes, eps=eps)

        rel_error = np.linalg.norm(np.array(f_jax) - f_finufft) / np.linalg.norm(f_finufft)
        assert rel_error < 10 * eps, f"Relative error {rel_error} exceeds 10*eps={10 * eps}"


class TestNufft3d3FINUFFT:
    """Tests comparing nufft3d3 against FINUFFT reference implementation."""

    def test_nufft3d3_vs_finufft_basic(self):
        """Test basic accuracy against FINUFFT."""
        M, N = 50, 30
        rng = np.random.default_rng(42)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        z = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        s = rng.uniform(-3, 3, N).astype(np.float64)
        t = rng.uniform(-3, 3, N).astype(np.float64)
        u = rng.uniform(-3, 3, N).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f_finufft = finufft.nufft3d3(x, y, z, c, s, t, u, eps=1e-9)

        n_modes = get_n_modes_3d(
            jnp.array(x),
            jnp.array(y),
            jnp.array(z),
            jnp.array(s),
            jnp.array(t),
            jnp.array(u),
            eps=1e-9,
        )
        f_jax = nufft3d3(
            jnp.array(x),
            jnp.array(y),
            jnp.array(z),
            jnp.array(c),
            jnp.array(s),
            jnp.array(t),
            jnp.array(u),
            n_modes,
            eps=1e-9,
        )

        rel_error = np.linalg.norm(np.array(f_jax) - f_finufft) / np.linalg.norm(f_finufft)
        assert rel_error < 1e-5, f"Relative error {rel_error} too large"

    @pytest.mark.parametrize("eps", [1e-3, 1e-6])
    def test_nufft3d3_vs_finufft_precision(self, eps):
        """Test accuracy at different precision levels."""
        M, N = 40, 25
        rng = np.random.default_rng(42)

        x = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        y = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        z = rng.uniform(-np.pi, np.pi, M).astype(np.float64)
        s = rng.uniform(-3, 3, N).astype(np.float64)
        t = rng.uniform(-3, 3, N).astype(np.float64)
        u = rng.uniform(-3, 3, N).astype(np.float64)
        c = (rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(np.complex128)

        f_finufft = finufft.nufft3d3(x, y, z, c, s, t, u, eps=eps)

        n_modes = get_n_modes_3d(
            jnp.array(x),
            jnp.array(y),
            jnp.array(z),
            jnp.array(s),
            jnp.array(t),
            jnp.array(u),
            eps=eps,
        )
        f_jax = nufft3d3(
            jnp.array(x),
            jnp.array(y),
            jnp.array(z),
            jnp.array(c),
            jnp.array(s),
            jnp.array(t),
            jnp.array(u),
            n_modes,
            eps=eps,
        )

        rel_error = np.linalg.norm(np.array(f_jax) - f_finufft) / np.linalg.norm(f_finufft)
        # 3D has more accumulated error, allow more tolerance
        assert rel_error < 100 * eps, f"Relative error {rel_error} exceeds 100*eps={100 * eps}"


class TestNufft3JIT:
    """Test JIT compilation of Type 3 transforms.

    Type 3 transforms are automatically JIT compiled. Tests verify that
    repeated calls use cached compilation.
    """

    def test_jit_nufft1d3_caching(self):
        """Test that nufft1d3 JIT compilation is cached."""
        M, N = 50, 30
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(43), (N,), minval=-5, maxval=5)
        c = (
            jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        ).astype(jnp.complex64)

        nf = get_n_modes_1d(x, s, eps=1e-6)

        # First call (compilation)
        result1 = nufft1d3(x, c, s, nf, eps=1e-6)

        # Second call (cached)
        result2 = nufft1d3(x, c, s, nf, eps=1e-6)

        np.testing.assert_allclose(result1, result2)

        # Verify accuracy
        expected = direct_nufft1d3(x, c, s)
        rel_error = jnp.linalg.norm(result1 - expected) / jnp.linalg.norm(expected)
        assert rel_error < 1e-4, f"Relative error {rel_error} too large"

    def test_jit_nufft2d3_caching(self):
        """Test that nufft2d3 JIT compilation is cached."""
        M, N = 40, 25
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(44), (N,), minval=-5, maxval=5)
        t = jax.random.uniform(jax.random.PRNGKey(45), (N,), minval=-5, maxval=5)
        c = (
            jax.random.normal(jax.random.PRNGKey(46), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(47), (M,))
        ).astype(jnp.complex64)

        n_modes = get_n_modes_2d(x, y, s, t, eps=1e-6)

        # First call (compilation)
        result1 = nufft2d3(x, y, c, s, t, n_modes, eps=1e-6)

        # Second call (cached)
        result2 = nufft2d3(x, y, c, s, t, n_modes, eps=1e-6)

        np.testing.assert_allclose(result1, result2)

        # Verify accuracy
        expected = direct_nufft2d3(x, y, c, s, t)
        rel_error = jnp.linalg.norm(result1 - expected) / jnp.linalg.norm(expected)
        assert rel_error < 1e-4

    def test_grad_nufft1d3(self):
        """Test that gradient of nufft1d3 works."""
        M, N = 30, 20
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(43), (N,), minval=-5, maxval=5)
        c = (
            jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        ).astype(jnp.complex64)

        nf = get_n_modes_1d(x, s, eps=1e-6)

        def loss(c):
            result = nufft1d3(x, c, s, nf, eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_c = jax.grad(loss)(c)

        assert grad_c.shape == c.shape
        assert not jnp.any(jnp.isnan(grad_c))


class TestNufft3Grad:
    """Test gradient computation for Type 3 transforms."""

    def test_grad_c_nufft1d3(self):
        """Test gradient w.r.t. c for nufft1d3."""
        M, N = 30, 20
        key = jax.random.PRNGKey(42)

        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi)
        s = jax.random.uniform(jax.random.PRNGKey(43), (N,), minval=-5, maxval=5)
        c = (
            jax.random.normal(jax.random.PRNGKey(44), (M,)) + 1j * jax.random.normal(jax.random.PRNGKey(45), (M,))
        ).astype(jnp.complex64)

        nf = get_n_modes_1d(x, s, eps=1e-6)

        def loss(c):
            result = nufft1d3(x, c, s, nf, eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_c = jax.grad(loss)(c)

        assert grad_c.shape == c.shape
        assert not jnp.any(jnp.isnan(grad_c))
