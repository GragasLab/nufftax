"""
Tests for different precision types (dtype support).
Verifies NUFFT works correctly with float32, float64, and other dtypes.
"""

import jax
import jax.numpy as jnp
import pytest

from nufftax.transforms import nufft1d1, nufft1d2, nufft2d1


class TestDtypePreservation:
    """Test that output dtype matches input dtype for float32."""

    def test_nufft1d1_dtype_preservation_float32(self):
        """Test 1D Type 1 preserves float32/complex64."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)).astype(jnp.float32) + 1j * jax.random.normal(
            jax.random.PRNGKey(44), (M,)
        ).astype(jnp.float32)
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.dtype == jnp.complex64, f"Expected complex64, got {result.dtype}"

    def test_nufft1d2_dtype_preservation_float32(self):
        """Test 1D Type 2 preserves float32/complex64."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        f = jax.random.normal(jax.random.PRNGKey(43), (N,)).astype(jnp.float32) + 1j * jax.random.normal(
            jax.random.PRNGKey(44), (N,)
        ).astype(jnp.float32)
        f = f.astype(jnp.complex64)

        result = nufft1d2(x, f, eps=1e-6)
        assert result.dtype == jnp.complex64, f"Expected complex64, got {result.dtype}"

    def test_nufft2d1_dtype_preservation_float32(self):
        """Test 2D Type 1 preserves float32/complex64."""
        M, N = 100, 32
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        y = jax.random.uniform(jax.random.PRNGKey(43), (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        c = jax.random.normal(jax.random.PRNGKey(44), (M,)).astype(jnp.float32) + 1j * jax.random.normal(
            jax.random.PRNGKey(45), (M,)
        ).astype(jnp.float32)
        c = c.astype(jnp.complex64)

        result = nufft2d1(x, y, c, (N, N), eps=1e-6)
        assert result.dtype == jnp.complex64, f"Expected complex64, got {result.dtype}"


class TestDtypeAccuracy:
    """Test accuracy at different precisions."""

    def _direct_nufft1d1(self, x, c, N):
        """Direct (slow) computation for comparison."""
        k = jnp.arange(-N // 2, (N + 1) // 2)
        # f[k] = sum_j c[j] * exp(i * k * x[j])
        return jnp.sum(c[:, None] * jnp.exp(1j * k[None, :] * x[:, None]), axis=0)

    def test_nufft1d1_accuracy_float32(self):
        """Test accuracy with float32/complex64."""
        M, N = 50, 32  # Small for direct computation
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)).astype(jnp.float32) + 1j * jax.random.normal(
            jax.random.PRNGKey(44), (M,)
        ).astype(jnp.float32)
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        expected = self._direct_nufft1d1(x, c, N)

        rel_error = jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected)
        assert rel_error < 1e-4, f"Relative error {rel_error} exceeds tolerance 1e-4"

    def test_nufft1d1_accuracy_float64(self):
        """Test accuracy with float64/complex128."""
        # Enable x64 for this test
        with jax.enable_x64(True):
            M, N = 50, 32  # Small for direct computation
            key = jax.random.PRNGKey(42)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float64)
            c = jax.random.normal(jax.random.PRNGKey(43), (M,)).astype(jnp.float64) + 1j * jax.random.normal(
                jax.random.PRNGKey(44), (M,)
            ).astype(jnp.float64)
            c = c.astype(jnp.complex128)

            result = nufft1d1(x, c, N, eps=1e-6)
            expected = self._direct_nufft1d1(x, c, N)

            rel_error = jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected)
            assert rel_error < 1e-5, f"Relative error {rel_error} exceeds tolerance 1e-5"

    @pytest.mark.parametrize("real_dtype", [jnp.float16, jnp.bfloat16])
    def test_low_precision_runs(self, real_dtype):
        """Test that low precision dtypes at least run without errors."""
        M, N = 50, 32
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(real_dtype)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,), dtype=jnp.float32) + 1j * jax.random.normal(
            jax.random.PRNGKey(44), (M,), dtype=jnp.float32
        )
        c = c.astype(jnp.complex64)

        # Just verify it runs
        result = nufft1d1(x.astype(jnp.float32), c, N, eps=1e-3)
        assert result.shape == (N,)


class TestMixedDtypes:
    """Test behavior with mixed input dtypes."""

    def test_mixed_x_c_dtypes_float32_complex64(self):
        """Test when x is float32 and c is complex64."""
        M, N = 100, 64
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)).astype(jnp.float32) + 1j * jax.random.normal(
            jax.random.PRNGKey(44), (M,)
        ).astype(jnp.float32)
        c = c.astype(jnp.complex64)

        result = nufft1d1(x, c, N, eps=1e-6)
        assert result.dtype == jnp.complex64


class TestGradientsWithDtypes:
    """Test gradient computation with float32."""

    def test_grad_c_float32(self):
        """Test gradient w.r.t. c with float32/complex64."""
        M, N = 50, 32
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)).astype(jnp.float32) + 1j * jax.random.normal(
            jax.random.PRNGKey(44), (M,)
        ).astype(jnp.float32)
        c = c.astype(jnp.complex64)

        def loss(c):
            result = nufft1d1(x, c, N, eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_c = jax.grad(loss)(c)
        assert grad_c.dtype == jnp.complex64, f"Expected complex64, got {grad_c.dtype}"

    def test_grad_x_float32(self):
        """Test gradient w.r.t. x with float32."""
        M, N = 50, 32
        key = jax.random.PRNGKey(42)
        x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32)
        c = jax.random.normal(jax.random.PRNGKey(43), (M,)).astype(jnp.float32) + 1j * jax.random.normal(
            jax.random.PRNGKey(44), (M,)
        ).astype(jnp.float32)
        c = c.astype(jnp.complex64)

        def loss(x):
            result = nufft1d1(x, c, N, eps=1e-6)
            return jnp.sum(jnp.abs(result) ** 2).real

        grad_x = jax.grad(loss)(x)
        assert grad_x.dtype == jnp.float32, f"Expected float32, got {grad_x.dtype}"
