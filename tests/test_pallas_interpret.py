"""Pallas spreading kernels in interpret mode (CPU, no GPU needed).

Run with ``NUFFTAX_PALLAS_INTERPRET=1`` (a dedicated CI step does). This
exercises the Pallas kernel logic — fold/rescale, indexing, kernel weights,
float32 casts, the dtype dispatch guard — without Triton or a GPU.

Interpret mode does NOT emulate atomic accumulation for duplicate indices, so
all tests here use collision-free points: x-coordinates spaced more than
nspread grid cells apart, which makes every kernel footprint disjoint (the 2D/3D
footprints are separable, so disjoint x-cells suffice).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nufftax.core.kernel import Kernel, compute_kernel_params
from nufftax.core.pallas_spread import INTERPRET, spread_1d_pallas, spread_2d_pallas, spread_3d_pallas
from nufftax.core.spread import spread_1d, spread_1d_impl, spread_2d_impl, spread_3d_impl


pytestmark = pytest.mark.skipif(not INTERPRET, reason="run with NUFFTAX_PALLAS_INTERPRET=1")

KP = compute_kernel_params(1e-6)  # nspread = 8
M = 24
NF1 = 512  # large enough for M footprints spaced 2*nspread apart, away from wrap


def _collision_free_x(rng):
    # Grid cells 2*nspread apart -> footprints [i0, i0+nspread-1] are disjoint
    # and never wrap (max cell well below NF1).
    cells = 2 * KP.nspread * np.arange(M) + KP.nspread + rng.uniform(0.1, 0.9, M)
    return jnp.asarray((cells / NF1) * 2.0 * np.pi - np.pi, dtype=jnp.float32)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def x(rng):
    return _collision_free_x(rng)


@pytest.fixture
def c(rng):
    return jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(jnp.complex64)


def test_spread_1d_interpret_matches_impl(x, c):
    out = spread_1d_pallas(x, c, NF1, KP)
    ref = spread_1d_impl(x, c, NF1, KP)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=5e-4, atol=5e-5)


def test_spread_2d_interpret_matches_impl(x, c, rng):
    y = jnp.asarray(rng.uniform(-np.pi, np.pi, M), dtype=jnp.float32)
    out = spread_2d_pallas(x, y, c, NF1, 32, KP)
    ref = spread_2d_impl(x, y, c, NF1, 32, KP)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=5e-4, atol=5e-5)


def test_spread_3d_interpret_matches_impl(x, c, rng):
    y = jnp.asarray(rng.uniform(-np.pi, np.pi, M), dtype=jnp.float32)
    z = jnp.asarray(rng.uniform(-np.pi, np.pi, M), dtype=jnp.float32)
    out = spread_3d_pallas(x, y, z, c, NF1, 16, 16, KP)
    ref = spread_3d_impl(x, y, z, c, NF1, 16, 16, KP)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=5e-4, atol=5e-5)


def test_custom_kernel_interpret_matches_impl(x, c):
    sigma = 1.5
    kernel = Kernel(nspread=KP.nspread, phi=lambda zz: jnp.exp(-(zz * zz) / (2.0 * sigma * sigma)))
    out = spread_1d_pallas(x, c, NF1, kernel)
    ref = spread_1d_impl(x, c, NF1, kernel)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=5e-4, atol=5e-5)


def test_dispatch_routes_32bit_to_pallas(x, c):
    # With interpret enabled, the public spread_1d routes 32-bit inputs through
    # the Pallas path; on collision-free points it must match pure JAX.
    out = spread_1d(x, c, NF1, KP)
    ref = spread_1d_impl(x, c, NF1, KP)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=5e-4, atol=5e-5)


def test_dispatch_keeps_64bit_on_pure_jax(rng):
    # Regression test for the x64 routing bug: 64-bit inputs must NOT go
    # through the float32 Pallas kernels. With float64/complex128 inputs the
    # public spread_1d must match the pure-JAX impl to ~1e-12; if it were
    # routed through Pallas the float32 cast would cap accuracy around 1e-7.
    assert jax.config.jax_enable_x64  # suite-wide via conftest
    x64 = jnp.asarray(_collision_free_x(rng), dtype=jnp.float64)
    c128 = jnp.asarray(rng.standard_normal(M) + 1j * rng.standard_normal(M)).astype(jnp.complex128)
    out = spread_1d(x64, c128, NF1, KP)
    ref = spread_1d_impl(x64, c128, NF1, KP)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-12, atol=1e-12)
