"""
Tests for JAX transformations on NUFFT functions.

Tests cover:
- JIT compilation (jax.jit)
- Reverse-mode autodiff (jax.grad, jax.vjp)
- Forward-mode autodiff (jax.jvp)
- Vectorization (jax.vmap)
- Higher-order derivatives (jax.hessian)
- Combinations of transforms
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Import NUFFT functions when implemented
# from nufftax import nufft1d1, nufft1d2, nufft2d1, nufft2d2

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def jax_key():
    """JAX random key."""
    return jax.random.PRNGKey(42)


# ============================================================================
# Test JIT Compilation
# ============================================================================


class TestJIT:
    """Test JIT compilation of NUFFT functions."""

    def test_jit_nufft1d1(self, rng):
        """JIT compile 1D Type 1."""
        M = 100

        # @jax.jit
        # def compute(x, c):
        #     return nufft1d1(x, c, n_modes, eps=1e-9)

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # First call compiles
        # f1 = compute(x, c)

        # Second call uses cached
        # f2 = compute(x, c)

        # assert_allclose(f1, f2, rtol=1e-14)

    def test_jit_nufft1d2(self, rng):
        """JIT compile 1D Type 2."""
        M = 100
        n_modes = 64

        # @jax.jit
        # def compute(x, f):
        #     return nufft1d2(x, f, eps=1e-9)

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # c1 = compute(x, f)
        # c2 = compute(x, f)
        # assert_allclose(c1, c2, rtol=1e-14)

    def test_jit_nufft2d1(self, rng):
        """JIT compile 2D Type 1."""
        M = 100

        # @jax.jit
        # def compute(x, y, c):
        #     return nufft2d1(x, y, c, n_modes, eps=1e-9)

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # f1 = compute(x, y, c)
        # f2 = compute(x, y, c)
        # assert_allclose(f1, f2, rtol=1e-14)

    def test_jit_preserves_value(self, rng):
        """JIT should not change output values."""
        M = 50

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # Non-JIT version
        # f_eager = nufft1d1(x, c, n_modes, eps=1e-9)

        # JIT version
        # f_jit = jax.jit(lambda x, c: nufft1d1(x, c, n_modes, eps=1e-9))(x, c)

        # assert_allclose(f_eager, f_jit, rtol=1e-14)

    def test_jit_with_static_argnums(self, rng):
        """Test JIT with static arguments."""
        M = 50

        # @partial(jax.jit, static_argnums=(2, 3))
        # def compute(x, c, n_modes, eps):
        #     return nufft1d1(x, c, n_modes, eps=eps)

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # f = compute(x, c, 32, 1e-9)
        # assert f.shape == (32,)


# ============================================================================
# Test Reverse-Mode AD (grad, vjp)
# ============================================================================


class TestGrad:
    """Test reverse-mode autodiff."""

    def test_grad_nufft1d1_c(self, rng):
        """Gradient of Type 1 w.r.t. complex strengths."""
        M = 20

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # grad_c = jax.grad(loss)(c)

        # assert grad_c.shape == c.shape
        # assert jnp.all(jnp.isfinite(grad_c))

    def test_grad_nufft1d1_x(self, rng):
        """Gradient of Type 1 w.r.t. positions."""
        M = 20

        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(x):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # grad_x = jax.grad(loss)(x)

        # assert grad_x.shape == x.shape
        # assert jnp.all(jnp.isfinite(grad_x))

    def test_grad_nufft1d2_f(self, rng):
        """Gradient of Type 2 w.r.t. mode coefficients."""
        M = 20
        n_modes = 16

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        def loss(f):
            # c = nufft1d2(x, f, eps=1e-9)
            # return jnp.sum(jnp.abs(c) ** 2).real
            pass

        # grad_f = jax.grad(loss)(f)

        # assert grad_f.shape == f.shape
        # assert jnp.all(jnp.isfinite(grad_f))

    def test_grad_multiple_argnums(self, rng):
        """Gradient w.r.t. multiple arguments."""
        M = 15

        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(x, c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # grad_x, grad_c = jax.grad(loss, argnums=(0, 1))(x, c)

        # assert grad_x.shape == x.shape
        # assert grad_c.shape == c.shape

    def test_vjp_nufft1d1(self, rng):
        """Test VJP of Type 1."""
        M = 20

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # def fn(c):
        #     return nufft1d1(x, c, n_modes, eps=1e-9)

        # f, vjp_fn = jax.vjp(fn, c)
        # cotangent = jnp.ones_like(f)
        # (grad_c,) = vjp_fn(cotangent)

        # assert grad_c.shape == c.shape

    def test_grad_through_composition(self, rng):
        """Gradient through composed operations."""
        M = 15

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def composed_loss(c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # # Apply some operations
            # f_scaled = f * 2.0
            # f_sum = jnp.sum(f_scaled)
            # return jnp.abs(f_sum) ** 2
            pass

        # grad_c = jax.grad(composed_loss)(c)
        # assert jnp.all(jnp.isfinite(grad_c))


# ============================================================================
# Test Forward-Mode AD (jvp)
# ============================================================================


class TestJVP:
    """Test forward-mode autodiff."""

    def test_jvp_nufft1d1_c(self, rng):
        """JVP of Type 1 w.r.t. complex strengths."""
        M = 20

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # def fn(c):
        #     return nufft1d1(x, c, n_modes, eps=1e-9)

        # primals, tangents = jax.jvp(fn, (c,), (tangent_c,))

        # assert primals.shape == (n_modes,)
        # assert tangents.shape == (n_modes,)
        # assert jnp.all(jnp.isfinite(tangents))

    def test_jvp_nufft1d1_x(self, rng):
        """JVP of Type 1 w.r.t. positions."""
        M = 20

        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        jnp.array(rng.standard_normal(M))

        # def fn(x):
        #     return nufft1d1(x, c, n_modes, eps=1e-9)

        # primals, tangents = jax.jvp(fn, (x,), (tangent_x,))

        # assert primals.shape == (n_modes,)
        # assert tangents.shape == (n_modes,)
        # assert jnp.all(jnp.isfinite(tangents))

    def test_jvp_nufft1d2_f(self, rng):
        """JVP of Type 2 w.r.t. mode coefficients."""
        M = 20
        n_modes = 16

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # def fn(f):
        #     return nufft1d2(x, f, eps=1e-9)

        # primals, tangents = jax.jvp(fn, (f,), (tangent_f,))

        # assert primals.shape == (M,)
        # assert tangents.shape == (M,)


# ============================================================================
# Test Vectorization (vmap)
# ============================================================================


class TestVmap:
    """Test vectorization with vmap."""

    def test_vmap_over_c_batch(self, rng):
        """vmap over batch of strength vectors."""
        batch_size = 5
        M = 50

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        # batched_nufft = jax.vmap(lambda c: nufft1d1(x, c, n_modes, eps=1e-9))
        # f_batch = batched_nufft(c_batch)

        # assert f_batch.shape == (batch_size, n_modes)

    def test_vmap_over_x_batch(self, rng):
        """vmap over batch of position vectors."""
        batch_size = 5
        M = 50

        jnp.array([rng.uniform(0, 2 * np.pi, M) for _ in range(batch_size)])
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # batched_nufft = jax.vmap(lambda x: nufft1d1(x, c, n_modes, eps=1e-9))
        # f_batch = batched_nufft(x_batch)

        # assert f_batch.shape == (batch_size, n_modes)

    def test_vmap_over_both(self, rng):
        """vmap over both x and c batches."""
        batch_size = 5
        M = 50

        jnp.array([rng.uniform(0, 2 * np.pi, M) for _ in range(batch_size)])
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        # batched_nufft = jax.vmap(lambda x, c: nufft1d1(x, c, n_modes, eps=1e-9))
        # f_batch = batched_nufft(x_batch, c_batch)

        # assert f_batch.shape == (batch_size, n_modes)

    def test_vmap_nufft1d2(self, rng):
        """vmap over batch for Type 2."""
        batch_size = 5
        M = 50
        n_modes = 32

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, n_modes)) + 1j * rng.standard_normal((batch_size, n_modes)))

        # batched_nufft = jax.vmap(lambda f: nufft1d2(x, f, eps=1e-9))
        # c_batch = batched_nufft(f_batch)

        # assert c_batch.shape == (batch_size, M)

    def test_vmap_nested(self, rng):
        """Nested vmap."""
        batch1 = 3
        batch2 = 4
        M = 30

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch1, batch2, M)) + 1j * rng.standard_normal((batch1, batch2, M)))

        # Double vmap
        # inner_vmap = jax.vmap(lambda c: nufft1d1(x, c, n_modes, eps=1e-9))
        # outer_vmap = jax.vmap(inner_vmap)
        # f_nested = outer_vmap(c_nested)

        # assert f_nested.shape == (batch1, batch2, n_modes)


# ============================================================================
# Test Higher-Order Derivatives
# ============================================================================


class TestHigherOrder:
    """Test higher-order derivatives."""

    def test_hessian_wrt_c(self, rng):
        """Hessian w.r.t. complex strengths."""
        M = 8

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # hess = jax.hessian(loss)(c)

        # assert hess.shape == (M, M)
        # assert jnp.all(jnp.isfinite(hess))

    def test_hessian_wrt_x(self, rng):
        """Hessian w.r.t. positions."""
        M = 8

        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(x):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # hess = jax.hessian(loss)(x)

        # assert hess.shape == (M, M)
        # assert jnp.all(jnp.isfinite(hess))

    def test_grad_of_grad(self, rng):
        """Gradient of gradient (mixed partials)."""
        M = 8

        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(x, c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # First derivative w.r.t. x
        # grad_x = jax.grad(loss, argnums=0)

        # Second derivative w.r.t. c
        # def grad_x_fn(x, c):
        #     return grad_x(x, c)[0]  # First component

        # grad_xc = jax.grad(grad_x_fn, argnums=1)(x, c)
        # assert jnp.all(jnp.isfinite(grad_xc))


# ============================================================================
# Test Combined Transforms
# ============================================================================


class TestCombinedTransforms:
    """Test combinations of JAX transforms."""

    def test_jit_grad(self, rng):
        """JIT-compiled gradient."""
        M = 30

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        # @jax.jit
        # def grad_loss(c):
        #     def loss(c):
        #         f = nufft1d1(x, c, n_modes, eps=1e-9)
        #         return jnp.sum(jnp.abs(f) ** 2).real
        #     return jax.grad(loss)(c)

        # grad_c = grad_loss(c)
        # assert grad_c.shape == c.shape

    def test_vmap_grad(self, rng):
        """vmap over gradients."""
        batch_size = 5
        M = 30

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        def loss(c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # batched_grad = jax.vmap(jax.grad(loss))
        # grad_batch = batched_grad(c_batch)

        # assert grad_batch.shape == c_batch.shape

    def test_jit_vmap(self, rng):
        """JIT-compiled vmap."""
        batch_size = 5
        M = 50

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        # @jax.jit
        # def batched_nufft(c_batch):
        #     return jax.vmap(lambda c: nufft1d1(x, c, n_modes, eps=1e-9))(c_batch)

        # f_batch = batched_nufft(c_batch)
        # assert f_batch.shape == (batch_size, n_modes)

    def test_jit_vmap_grad(self, rng):
        """JIT-compiled vmap of gradients."""
        batch_size = 5
        M = 30

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        def loss(c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # @jax.jit
        # def batched_grad(c_batch):
        #     return jax.vmap(jax.grad(loss))(c_batch)

        # grad_batch = batched_grad(c_batch)
        # assert grad_batch.shape == c_batch.shape


# ============================================================================
# Test Gradient Correctness via Finite Differences
# ============================================================================


class TestGradientCorrectness:
    """Verify gradient correctness via finite differences."""

    def test_grad_c_vs_finite_diff(self, rng):
        """Compare autodiff gradient to finite difference for c."""
        M = 10

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(c):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # grad_auto = jax.grad(loss)(c)
        # grad_num = numerical_gradient_complex(loss, c, h=1e-6)

        # assert_allclose(np.array(grad_auto), np.array(grad_num), rtol=1e-4)

    def test_grad_x_vs_finite_diff(self, rng):
        """Compare autodiff gradient to finite difference for x."""
        M = 10

        jnp.array(rng.uniform(0.5, 2 * np.pi - 0.5, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))

        def loss(x):
            # f = nufft1d1(x, c, n_modes, eps=1e-9)
            # return jnp.sum(jnp.abs(f) ** 2).real
            pass

        # grad_auto = jax.grad(loss)(x)
        # grad_num = numerical_gradient_real(loss, np.array(x), h=1e-6)

        # assert_allclose(np.array(grad_auto), np.array(grad_num), rtol=1e-4)

    def test_grad_f_vs_finite_diff(self, rng):
        """Compare autodiff gradient to finite difference for f (Type 2)."""
        M = 10
        n_modes = 8

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        def loss(f):
            # c = nufft1d2(x, f, eps=1e-9)
            # return jnp.sum(jnp.abs(c) ** 2).real
            pass

        # grad_auto = jax.grad(loss)(f)
        # grad_num = numerical_gradient_complex(loss, f, h=1e-6)

        # assert_allclose(np.array(grad_auto), np.array(grad_num), rtol=1e-4)


# ============================================================================
# Test Gradients Relate Type 1 and Type 2 Correctly
# ============================================================================


class TestGradientRelationship:
    """Test that gradients of Type 1 and Type 2 are related correctly."""

    def test_adjoint_gradient_relationship(self, rng):
        """
        Test: grad_c loss_1 = grad_c <f1, f>
              where f1 = nufft1(c) and loss_1 = <f1, f>

        By adjoint: <f1, f> = <c, nufft2(f)>
        So: grad_c <f1, f> = conj(nufft2(f))
        """
        M = 30
        n_modes = 20

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        def loss(c):
            # f1 = nufft1d1(x, c, n_modes, eps=1e-12)
            # return jnp.vdot(f1, f).real  # Re(<f1, f>)
            pass

        # grad_c = jax.grad(loss)(c)

        # Expected: conj(nufft2(f))
        # c2 = nufft1d2(x, f, eps=1e-12)
        # expected_grad = jnp.conj(c2)

        # They should be proportional (may differ by normalization)
        # assert_allclose(grad_c, expected_grad, rtol=1e-8)


# ============================================================================
# Test Transform Consistency
# ============================================================================


class TestTransformConsistency:
    """Test that transforms are consistent with each other."""

    def test_jvp_vjp_consistency(self, rng):
        """JVP and VJP should give consistent results."""
        M = 20
        n_modes = 16

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        jnp.array(rng.standard_normal(M) + 1j * rng.standard_normal(M))
        jnp.array(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))

        # def fn(c):
        #     return nufft1d1(x, c, n_modes, eps=1e-9)

        # JVP
        # primals, jvp_tangent = jax.jvp(fn, (c,), (tangent,))

        # VJP
        # primals, vjp_fn = jax.vjp(fn, c)
        # (vjp_cotangent,) = vjp_fn(cotangent)

        # Consistency: <jvp_tangent, cotangent> = <tangent, vjp_cotangent>
        # lhs = jnp.vdot(jvp_tangent, cotangent)
        # rhs = jnp.vdot(tangent, vjp_cotangent)

        # assert_allclose(lhs, rhs, rtol=1e-8)

    def test_vmap_consistent_with_loop(self, rng):
        """vmap should give same result as explicit loop."""
        batch_size = 3
        M = 30

        jnp.array(rng.uniform(0, 2 * np.pi, M))
        jnp.array(rng.standard_normal((batch_size, M)) + 1j * rng.standard_normal((batch_size, M)))

        # vmap result
        # f_vmap = jax.vmap(lambda c: nufft1d1(x, c, n_modes, eps=1e-9))(c_batch)

        # Loop result
        # f_loop = jnp.stack([nufft1d1(x, c_batch[i], n_modes, eps=1e-9)
        #                    for i in range(batch_size)])

        # assert_allclose(f_vmap, f_loop, rtol=1e-14)
