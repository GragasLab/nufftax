"""Tests covering jax.linear_transpose and lax.custom_linear_solve compatibility.

Regression coverage for issue #5: previously, the public NUFFT functions were
wrapped in @jax.custom_vjp, whose underlying primitive (custom_vjp_call) has no
transpose rule. This broke jax.scipy.sparse.linalg.{cg,gmres,bicgstab} which
internally call lax.custom_linear_solve -> jax.linear_transpose.

With the primitive-based implementation, transpose rules are explicit:
- nufft1d1 <-> nufft1d2 (same isign, linear transpose)
- nufft2d1 <-> nufft2d2
- nufft3d1 <-> nufft3d2
- Type 3 is self-adjoint with source/target points swapped.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from nufftax import nufft1d1, nufft1d2, nufft2d1, nufft2d2


class TestLinearTranspose:
    def test_1d_round_trip(self):
        """linear_transpose(nufft1d1)(g) should equal nufft1d2(x, g) with same isign."""
        M, N = 32, 64
        x = jnp.linspace(-jnp.pi, jnp.pi, M).astype(jnp.float32)
        c = jnp.ones(M, dtype=jnp.complex64)
        g = jnp.arange(N, dtype=jnp.complex64) + 1j

        lt = jax.linear_transpose(lambda v: nufft1d1(x, v, N, eps=1e-8, isign=1), c)
        (got,) = lt(g)
        want = nufft1d2(x, g, eps=1e-8, isign=1)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_2d_round_trip(self):
        n1, n2, M = 16, 12, 50
        x = jnp.linspace(-jnp.pi, jnp.pi, M).astype(jnp.float32)
        y = jnp.linspace(-jnp.pi, jnp.pi, M).astype(jnp.float32)
        c = jnp.ones(M, dtype=jnp.complex64)
        g = jnp.ones((n2, n1), dtype=jnp.complex64)

        lt = jax.linear_transpose(lambda v: nufft2d1(x, y, v, (n1, n2), eps=1e-8, isign=1), c)
        (got,) = lt(g)
        want = nufft2d2(x, y, g, eps=1e-8, isign=1)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


class TestConjugateGradient:
    def test_issue_5_repro(self):
        """Exact repro of the failing example posted in issue #5."""
        n = 32
        m = 256
        x = jnp.linspace(-3.14, 3.14, m).astype(jnp.float32)
        y = jnp.zeros(m, dtype=jnp.float32)

        def forward(flat_img):
            img = flat_img.reshape(n, n).astype(jnp.complex64)
            return nufft2d2(x, y, img, eps=1e-6)

        def adjoint(kspace):
            return nufft2d1(x, y, kspace, (n, n), eps=1e-6).ravel()

        def symmetric_op(v):
            return adjoint(forward(v))

        b = symmetric_op(jnp.ones(n * n, dtype=jnp.complex64))
        result, _ = cg(symmetric_op, b, x0=jnp.zeros(n * n, dtype=jnp.complex64), maxiter=5)
        assert result.shape == (n * n,)
        assert not jnp.any(jnp.isnan(result))

    def test_cg_converges_on_hermitian_system(self):
        """Solve A^H A f = A^H b and verify CG actually converges.

        nufft1d2 with isign=-1 and nufft1d1 with isign=+1 are Hermitian adjoints
        of each other, so A^H A is Hermitian PSD and CG is well-defined.
        """
        with jax.enable_x64(True):
            M, N = 64, 32
            key = jax.random.PRNGKey(0)
            x = jax.random.uniform(key, (M,), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64)
            f_true = jax.random.normal(jax.random.PRNGKey(1), (N,), dtype=jnp.float64) + 1j * jax.random.normal(
                jax.random.PRNGKey(2), (N,), dtype=jnp.float64
            )

            def A(f):
                return nufft1d2(x, f, eps=1e-10, isign=-1)

            def Ah(c):
                return nufft1d1(x, c, N, eps=1e-10, isign=+1)

            b = A(f_true)
            f_solved, _ = cg(lambda f: Ah(A(f)), Ah(b), maxiter=200, tol=1e-10)

            residual = jnp.linalg.norm(A(f_solved) - b) / jnp.linalg.norm(b)
            assert residual < 1e-3, f"CG did not converge: relative residual = {residual}"
