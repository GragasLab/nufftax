"""Regression tests for the batched complex64 scatter used by the pure-JAX spread.

``jax.ops.segment_sum`` on a batched *complex64* array is memory-unsafe on some
JAX/GPU combinations above roughly ``2**18`` updates: the generated kernel writes
out of bounds, which showed up as exact zeros for trailing batch elements (or as
``CUDA_ERROR_ILLEGAL_ADDRESS``). The float32, complex128 and CPU versions of the
same call are exact. ``spread_*_impl`` now accumulates the real and imaginary
parts separately (``_segment_sum_complex``), so these tests pin both the
primitive and the user-visible transform.

Measured on jax 0.11.2 / jaxlib 0.11.2 / H100 PCIe (driver 580.159.03), with the
Pallas backend off:

    grid  m     C   updates   before      after
    256   5120  4   1310720   9.33e-01    3.12e-07
    256   1024  4   262144    1.00e+00    2.14e-07   (zeros for rows 2, 3 before)
    256   5120  8   2621440   1.42e+00    2.45e-07

The tests do not assert that the *old* path is broken -- whether the
out-of-bounds write corrupts a given result depends on the allocation layout --
they assert that the spreading path is now correct at the sizes where it was not.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _relative_error(a: jax.Array, b: jax.Array) -> float:
    """Max-abs difference relative to the magnitude of ``b``.

    Element-wise comparison with a small absolute tolerance is too strict here:
    the batched and per-row accumulations add in a different order, so a
    near-cancelling element (order 1e-2 against an array of order 1e2) can differ
    by more than a tiny ``atol`` while still being round-off relative to the
    transform's scale. A broken path is off by 0.3-1.4, so this separates the two
    cleanly.
    """
    return float(jnp.abs(a - b).max() / jnp.abs(b).max())


def _split_accumulate(values: jax.Array, indices: jax.Array, num_segments: int) -> jax.Array:
    """``_segment_sum_complex``, imported lazily.

    The import is inside the function so that this module still imports on a tree
    without the fix: the integration test below then fails on the actual symptom
    (exact zeros for batch elements) rather than on a collection error.
    """
    from nufftax.core.spread import _segment_sum_complex

    return _segment_sum_complex(values, indices, num_segments)


def _samples(spokes: int, readout: int) -> tuple[jax.Array, jax.Array]:
    """Golden-angle radial sample coordinates in [-pi, pi)."""
    angle = np.arange(spokes) * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.linspace(-np.pi, np.pi, readout, endpoint=False)
    x = (np.cos(angle)[:, None] * radius[None, :]).reshape(-1)
    y = (np.sin(angle)[:, None] * radius[None, :]).reshape(-1)
    return jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32)


class TestSegmentSumComplex:
    """``_segment_sum_complex`` must equal the elementwise accumulation."""

    @pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
    @pytest.mark.parametrize("n_trans", [1, 4, 16])
    def test_batched_matches_per_row(self, dtype: jnp.dtype, n_trans: int) -> None:
        """A batched call must equal the same call row by row, at the broken size."""
        rng = np.random.default_rng(0)
        num_segments = 2**18
        updates_per_row = 16384
        indices = jnp.asarray(rng.integers(0, num_segments, size=updates_per_row).astype(np.int32))
        values = jnp.asarray(
            (rng.normal(size=(n_trans, updates_per_row)) + 1j * rng.normal(size=(n_trans, updates_per_row))).astype(
                dtype
            )
        )

        def accumulate(row: jax.Array) -> jax.Array:
            return _split_accumulate(row, indices, num_segments)

        batched = jax.vmap(accumulate)(values)
        per_row = jnp.stack([accumulate(values[i]) for i in range(n_trans)])
        assert _relative_error(batched, per_row) < 1e-5

    def test_real_values_are_scattered_directly(self) -> None:
        """A real dtype keeps the plain scatter and is exact."""
        rng = np.random.default_rng(1)
        indices = jnp.asarray(rng.integers(0, 4096, size=8192).astype(np.int32))
        values = jnp.asarray(rng.normal(size=(8, 8192)).astype(np.float32))
        batched = jax.vmap(lambda row: _split_accumulate(row, indices, 4096))(values)
        per_row = jnp.stack([_split_accumulate(values[i], indices, 4096) for i in range(values.shape[0])])
        assert _relative_error(batched, per_row) < 1e-5

    def test_spread_never_scatters_complex_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deterministic guard for the fix, independent of the memory layout.

        The corruption only appears on some allocations, so this pins the
        *mechanism* instead: no complex array may reach ``jax.ops.segment_sum``
        from the spreading code.
        """
        from nufftax.core.kernel import compute_kernel_params
        from nufftax.core.spread import spread_2d_impl

        original = jax.ops.segment_sum
        seen: list[str] = []

        def guarded(values, indices, num_segments=None, **kwargs):
            seen.append(str(values.dtype))
            assert not jnp.issubdtype(values.dtype, jnp.complexfloating), (
                f"complex {values.dtype} was passed to segment_sum directly"
            )
            return original(values, indices, num_segments=num_segments, **kwargs)

        monkeypatch.setattr(jax.ops, "segment_sum", guarded)
        rng = np.random.default_rng(4)
        x = jnp.asarray(rng.uniform(-np.pi, np.pi, 256), jnp.float32)
        y = jnp.asarray(rng.uniform(-np.pi, np.pi, 256), jnp.float32)
        coeffs = jnp.asarray((rng.normal(size=(3, 256)) + 1j * rng.normal(size=(3, 256))).astype(np.complex64))
        out = spread_2d_impl(x, y, coeffs, 32, 32, compute_kernel_params(1e-6, 2.0))
        assert out.shape == (3, 32, 32)
        assert seen, "segment_sum was never called: this test no longer covers the spread"
        assert all(dtype in {"float32", "float64"} for dtype in seen), seen

    def test_matches_reference_accumulation(self) -> None:
        """Compare against a dense numpy accumulation on a small case."""
        rng = np.random.default_rng(2)
        indices = rng.integers(0, 32, size=200)
        values = (rng.normal(size=200) + 1j * rng.normal(size=200)).astype(np.complex64)
        expected = np.zeros(32, dtype=np.complex128)
        np.add.at(expected, indices, values.astype(np.complex128))
        got = _split_accumulate(jnp.asarray(values), jnp.asarray(indices.astype(np.int32)), 32)
        assert _relative_error(got, jnp.asarray(expected)) < 1e-5


class TestBatchedType1IsCorrect:
    """The user-visible regression: a batched type-1 transform vs a per-transform loop."""

    @pytest.mark.parametrize("n_coils", [4, 8])
    def test_nufft2d1_batched_matches_loop_at_large_grid(self, n_coils: int) -> None:
        """``vmap(nufft2d1)`` must match a per-transform loop (was exact zeros for rows 2,3)."""
        import nufftax

        spokes, readout, grid = 20, 256, 256
        sample_x, sample_y = _samples(spokes, readout)
        sample_len = spokes * readout
        rng = np.random.default_rng(0)
        data = jnp.asarray(
            (rng.normal(size=(n_coils, sample_len)) + 1j * rng.normal(size=(n_coils, sample_len))).astype(np.complex64)
        )

        def adjoint(coeffs: jax.Array) -> jax.Array:
            return nufftax.nufft2d1(sample_x, sample_y, coeffs, n_modes=(grid, grid), eps=1e-6)

        batched = jax.vmap(adjoint)(data)
        looped = jnp.stack([adjoint(data[c]) for c in range(n_coils)])

        # The old failure mode returned exact zeros for some batch elements.
        for c in range(n_coils):
            assert not bool(jnp.all(batched[c] == 0)), f"batch element {c} is exactly zero"
        # Broken path: 0.33-1.4 relative. Fixed: ~3e-07 (float32 round-off).
        assert _relative_error(batched, looped) < 1e-5
