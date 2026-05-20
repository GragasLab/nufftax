"""
Pallas GPU spreading kernels for NUFFT (Type 1, 1D/2D/3D).

Triton backend, atomic scatter-add. Fuses coordinate scaling, ES kernel
evaluation and the scatter into one GPU kernel, avoiding the
O(M * nspread^d) intermediate that pure JAX materializes.

Benchmarked vs pure-JAX spread on H100 (M >= 100K): ~17-55x (1D/2D),
~3-4x (3D). Interpolation (Type 2 gather) is intentionally left to XLA:
benchmarked it was at best parity and 2-4x slower in 3D, so no Pallas
interp kernels are provided.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton


BLOCK_SIZE = 256


# ============================================================================
# Shared kernel logic
# ============================================================================


def _fold_rescale(x, nf):
    """[-pi, pi) -> [0, nf)"""
    inv_2pi = 1.0 / (2.0 * jnp.pi)
    x_scaled = x * inv_2pi + 0.5
    return (x_scaled - jnp.floor(x_scaled)) * nf


def _eval_kernel_1d(i0, k, x_scaled, beta, c):
    """Evaluate ES kernel weight for offset k from i0."""
    z = (i0 + k).astype(x_scaled.dtype) - x_scaled
    arg = 1.0 - c * z * z
    return jnp.where(arg >= 0, jnp.exp(beta * (jnp.sqrt(jnp.maximum(arg, 0.0)) - 1.0)), 0.0)


# ============================================================================
# 1D Spreading
# ============================================================================


def _spread_1d_kernel(
    x_ref,
    c_real_ref,
    c_imag_ref,
    fw_real_in_ref,
    fw_imag_in_ref,
    fw_real_out_ref,
    fw_imag_out_ref,
    *,
    nf,
    nspread,
    beta,
    c,
):
    x = x_ref[:]
    cr, ci = c_real_ref[:], c_imag_ref[:]
    x_scaled = _fold_rescale(x, nf)
    i0 = jnp.ceil(x_scaled - nspread / 2.0).astype(jnp.int32)

    for k in range(nspread):
        idx = (i0 + k) % nf
        w = _eval_kernel_1d(i0, k, x_scaled, beta, c)
        pltriton.atomic_add(fw_real_out_ref, idx, cr * w)
        pltriton.atomic_add(fw_imag_out_ref, idx, ci * w)


def spread_1d_pallas(x, c, nf, kernel_params):
    """1D spreading using fused Pallas kernel with atomic scatter-add."""
    M = x.shape[0]
    M_pad = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    x_pad = jnp.pad(x.astype(jnp.float32), (0, M_pad - M))
    c_real_pad = jnp.pad(jnp.real(c).astype(jnp.float32), (0, M_pad - M))
    c_imag_pad = jnp.pad(jnp.imag(c).astype(jnp.float32), (0, M_pad - M))
    fw_real_init = jnp.zeros((nf,), dtype=jnp.float32)
    fw_imag_init = jnp.zeros((nf,), dtype=jnp.float32)

    kernel_fn = functools.partial(
        _spread_1d_kernel,
        nf=nf,
        nspread=kernel_params.nspread,
        beta=float(kernel_params.beta),
        c=float(kernel_params.c),
    )
    fw_real, fw_imag = pl.pallas_call(
        kernel_fn,
        grid=(M_pad // BLOCK_SIZE,),
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((nf,), lambda i: (0,)),
            pl.BlockSpec((nf,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((nf,), lambda i: (0,)),
            pl.BlockSpec((nf,), lambda i: (0,)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((nf,), jnp.float32),
            jax.ShapeDtypeStruct((nf,), jnp.float32),
        ],
        input_output_aliases={3: 0, 4: 1},
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
    )(x_pad, c_real_pad, c_imag_pad, fw_real_init, fw_imag_init)
    return (fw_real + 1j * fw_imag).astype(c.dtype)


# ============================================================================
# 2D Spreading
# ============================================================================


def _spread_2d_kernel(
    x_ref,
    y_ref,
    c_real_ref,
    c_imag_ref,
    fw_real_in_ref,
    fw_imag_in_ref,
    fw_real_out_ref,
    fw_imag_out_ref,
    *,
    nf1,
    nf2,
    nspread,
    beta,
    c,
):
    x, y = x_ref[:], y_ref[:]
    cr, ci = c_real_ref[:], c_imag_ref[:]
    x_scaled = _fold_rescale(x, nf1)
    y_scaled = _fold_rescale(y, nf2)
    i0_x = jnp.ceil(x_scaled - nspread / 2.0).astype(jnp.int32)
    i0_y = jnp.ceil(y_scaled - nspread / 2.0).astype(jnp.int32)

    # Precompute 1D kernel values (separable factorization)
    wx_vals, idx_x_vals = [], []
    for kx in range(nspread):
        idx_x_vals.append((i0_x + kx) % nf1)
        wx_vals.append(_eval_kernel_1d(i0_x, kx, x_scaled, beta, c))
    wy_vals, idy_vals = [], []
    for ky in range(nspread):
        idy_vals.append((i0_y + ky) % nf2)
        wy_vals.append(_eval_kernel_1d(i0_y, ky, y_scaled, beta, c))

    for ky in range(nspread):
        for kx in range(nspread):
            w2d = wy_vals[ky] * wx_vals[kx]
            flat_idx = idy_vals[ky] * nf1 + idx_x_vals[kx]
            pltriton.atomic_add(fw_real_out_ref, flat_idx, cr * w2d)
            pltriton.atomic_add(fw_imag_out_ref, flat_idx, ci * w2d)


def spread_2d_pallas(x, y, c, nf1, nf2, kernel_params):
    """2D spreading using fused Pallas kernel with atomic scatter-add."""
    M = x.shape[0]
    nf_total = nf1 * nf2
    M_pad = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    x_pad = jnp.pad(x.astype(jnp.float32), (0, M_pad - M))
    y_pad = jnp.pad(y.astype(jnp.float32), (0, M_pad - M))
    c_real_pad = jnp.pad(jnp.real(c).astype(jnp.float32), (0, M_pad - M))
    c_imag_pad = jnp.pad(jnp.imag(c).astype(jnp.float32), (0, M_pad - M))
    fw_real_init = jnp.zeros((nf_total,), dtype=jnp.float32)
    fw_imag_init = jnp.zeros((nf_total,), dtype=jnp.float32)

    kernel_fn = functools.partial(
        _spread_2d_kernel,
        nf1=nf1,
        nf2=nf2,
        nspread=kernel_params.nspread,
        beta=float(kernel_params.beta),
        c=float(kernel_params.c),
    )
    fw_real, fw_imag = pl.pallas_call(
        kernel_fn,
        grid=(M_pad // BLOCK_SIZE,),
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((nf_total,), lambda i: (0,)),
            pl.BlockSpec((nf_total,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((nf_total,), lambda i: (0,)),
            pl.BlockSpec((nf_total,), lambda i: (0,)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((nf_total,), jnp.float32),
            jax.ShapeDtypeStruct((nf_total,), jnp.float32),
        ],
        input_output_aliases={4: 0, 5: 1},
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
    )(x_pad, y_pad, c_real_pad, c_imag_pad, fw_real_init, fw_imag_init)
    return (fw_real + 1j * fw_imag).astype(c.dtype).reshape(nf2, nf1)


# ============================================================================
# 3D Spreading
# ============================================================================


def _spread_3d_kernel(
    x_ref,
    y_ref,
    z_ref,
    c_real_ref,
    c_imag_ref,
    fw_real_in_ref,
    fw_imag_in_ref,
    fw_real_out_ref,
    fw_imag_out_ref,
    *,
    nf1,
    nf2,
    nf3,
    nspread,
    beta,
    c,
):
    x, y, z = x_ref[:], y_ref[:], z_ref[:]
    cr, ci = c_real_ref[:], c_imag_ref[:]
    x_scaled = _fold_rescale(x, nf1)
    y_scaled = _fold_rescale(y, nf2)
    z_scaled = _fold_rescale(z, nf3)
    i0_x = jnp.ceil(x_scaled - nspread / 2.0).astype(jnp.int32)
    i0_y = jnp.ceil(y_scaled - nspread / 2.0).astype(jnp.int32)
    i0_z = jnp.ceil(z_scaled - nspread / 2.0).astype(jnp.int32)

    # Triton-side nested loops over the nspread^3 footprint. A Python triple
    # loop here would unroll into ~nspread^3 atomic_add ops and make Triton
    # compile times explode in 3D. Three nested fori_loops keep the IR small
    # (one body per axis) while preserving the separable factorization by
    # hoisting the z/y weights into the outer loop bodies.
    nf12 = nf1 * nf2

    def kz_body(kz, _):
        wz = _eval_kernel_1d(i0_z, kz, z_scaled, beta, c)
        base_z = ((i0_z + kz) % nf3) * nf12

        def ky_body(ky, _):
            wyz = wz * _eval_kernel_1d(i0_y, ky, y_scaled, beta, c)
            base_yz = base_z + ((i0_y + ky) % nf2) * nf1

            def kx_body(kx, _):
                w = wyz * _eval_kernel_1d(i0_x, kx, x_scaled, beta, c)
                flat_idx = base_yz + ((i0_x + kx) % nf1)
                pltriton.atomic_add(fw_real_out_ref, flat_idx, cr * w)
                pltriton.atomic_add(fw_imag_out_ref, flat_idx, ci * w)
                return _

            return jax.lax.fori_loop(0, nspread, kx_body, _)

        return jax.lax.fori_loop(0, nspread, ky_body, _)

    jax.lax.fori_loop(0, nspread, kz_body, jnp.int32(0))


def spread_3d_pallas(x, y, z, c, nf1, nf2, nf3, kernel_params):
    """3D spreading using fused Pallas kernel with atomic scatter-add."""
    M = x.shape[0]
    nf_total = nf1 * nf2 * nf3
    M_pad = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    x_pad = jnp.pad(x.astype(jnp.float32), (0, M_pad - M))
    y_pad = jnp.pad(y.astype(jnp.float32), (0, M_pad - M))
    z_pad = jnp.pad(z.astype(jnp.float32), (0, M_pad - M))
    c_real_pad = jnp.pad(jnp.real(c).astype(jnp.float32), (0, M_pad - M))
    c_imag_pad = jnp.pad(jnp.imag(c).astype(jnp.float32), (0, M_pad - M))
    fw_real_init = jnp.zeros((nf_total,), dtype=jnp.float32)
    fw_imag_init = jnp.zeros((nf_total,), dtype=jnp.float32)

    kernel_fn = functools.partial(
        _spread_3d_kernel,
        nf1=nf1,
        nf2=nf2,
        nf3=nf3,
        nspread=kernel_params.nspread,
        beta=float(kernel_params.beta),
        c=float(kernel_params.c),
    )
    fw_real, fw_imag = pl.pallas_call(
        kernel_fn,
        grid=(M_pad // BLOCK_SIZE,),
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((nf_total,), lambda i: (0,)),
            pl.BlockSpec((nf_total,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((nf_total,), lambda i: (0,)),
            pl.BlockSpec((nf_total,), lambda i: (0,)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((nf_total,), jnp.float32),
            jax.ShapeDtypeStruct((nf_total,), jnp.float32),
        ],
        input_output_aliases={5: 0, 6: 1},
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
    )(x_pad, y_pad, z_pad, c_real_pad, c_imag_pad, fw_real_init, fw_imag_init)
    return (fw_real + 1j * fw_imag).astype(c.dtype).reshape(nf3, nf2, nf1)
