"""
Pallas GPU kernels for NUFFT spreading and interpolation (1D, 2D).

Spreading (Type 1) uses the Triton backend with atomic scatter-add.
Interpolation (Type 2) is a gather operation (no atomics needed).

Fuses coordinate scaling, ES kernel evaluation, and scatter/gather
into single GPU kernels, avoiding the O(M * nspread^d) intermediate
tensors that pure JAX materializes.

Benchmarked speedups over pure JAX (spreading):
  A100: 1D 5-67x, 2D 2-3x (M >= 100K)
  H100: 1D 4-75x, 2D 2.7-3.2x (M >= 100K)
"""

import functools

import jax
import jax.numpy as jnp
from jax._src.core import Primitive, ShapedArray
from jax._src.interpreters import ad as _ad
from jax._src.interpreters import batching as _batching
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

from .spread import (
    _HAS_PALLAS_GPU,
    _PALLAS_MIN_M_INTERP,
    _PALLAS_MIN_M_SPREAD,
)
from .spread import (
    _spread_1d_dispatch as _s1_disp,
)
from .spread import (
    _spread_2d_dispatch as _s2_disp,
)
from .spread import (
    interp_1d_impl as _i1_impl,
)
from .spread import (
    interp_2d_impl as _i2_impl,
)
from .spread import (
    spread_1d_impl as _s1_impl,
)
from .spread import (
    spread_2d_impl as _s2_impl,
)


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
# 1D Interpolation
# ============================================================================


def _interp_1d_kernel(
    x_ref,
    fw_real_ref,
    fw_imag_ref,
    c_real_ref,
    c_imag_ref,
    *,
    nf,
    nspread,
    beta,
    c,
):
    x = x_ref[:]
    x_scaled = _fold_rescale(x, nf)
    i0 = jnp.ceil(x_scaled - nspread / 2.0).astype(jnp.int32)

    cr_acc = jnp.zeros_like(x)
    ci_acc = jnp.zeros_like(x)
    for k in range(nspread):
        idx = (i0 + k) % nf
        w = _eval_kernel_1d(i0, k, x_scaled, beta, c)
        cr_acc = cr_acc + fw_real_ref[idx] * w
        ci_acc = ci_acc + fw_imag_ref[idx] * w

    c_real_ref[:] = cr_acc
    c_imag_ref[:] = ci_acc


def interp_1d_pallas(x, fw, kernel_params):
    """1D interpolation using fused Pallas kernel (gather, no atomics)."""
    M = x.shape[0]
    nf = fw.shape[-1]
    M_pad = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    x_pad = jnp.pad(x.astype(jnp.float32), (0, M_pad - M))
    fw_real = jnp.real(fw).astype(jnp.float32)
    fw_imag = jnp.imag(fw).astype(jnp.float32)

    kernel_fn = functools.partial(
        _interp_1d_kernel,
        nf=nf,
        nspread=kernel_params.nspread,
        beta=float(kernel_params.beta),
        c=float(kernel_params.c),
    )
    c_real, c_imag = pl.pallas_call(
        kernel_fn,
        grid=(M_pad // BLOCK_SIZE,),
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((nf,), lambda i: (0,)),
            pl.BlockSpec((nf,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((M_pad,), jnp.float32),
            jax.ShapeDtypeStruct((M_pad,), jnp.float32),
        ],
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
    )(x_pad, fw_real, fw_imag)
    return (c_real[:M] + 1j * c_imag[:M]).astype(fw.dtype)


# ============================================================================
# 2D Interpolation
# ============================================================================


def _interp_2d_kernel(
    x_ref,
    y_ref,
    fw_real_ref,
    fw_imag_ref,
    c_real_ref,
    c_imag_ref,
    *,
    nf1,
    nf2,
    nspread,
    beta,
    c,
):
    x, y = x_ref[:], y_ref[:]
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

    cr_acc = jnp.zeros_like(x)
    ci_acc = jnp.zeros_like(x)
    for ky in range(nspread):
        for kx in range(nspread):
            w2d = wy_vals[ky] * wx_vals[kx]
            flat_idx = idy_vals[ky] * nf1 + idx_x_vals[kx]
            cr_acc = cr_acc + fw_real_ref[flat_idx] * w2d
            ci_acc = ci_acc + fw_imag_ref[flat_idx] * w2d

    c_real_ref[:] = cr_acc
    c_imag_ref[:] = ci_acc


def interp_2d_pallas(x, y, fw, kernel_params):
    """2D interpolation using fused Pallas kernel (gather, no atomics)."""
    M = x.shape[0]
    nf2, nf1 = fw.shape[-2], fw.shape[-1]
    nf_total = nf1 * nf2
    M_pad = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    x_pad = jnp.pad(x.astype(jnp.float32), (0, M_pad - M))
    y_pad = jnp.pad(y.astype(jnp.float32), (0, M_pad - M))
    fw_flat = fw.reshape(-1)
    fw_real = jnp.real(fw_flat).astype(jnp.float32)
    fw_imag = jnp.imag(fw_flat).astype(jnp.float32)

    kernel_fn = functools.partial(
        _interp_2d_kernel,
        nf1=nf1,
        nf2=nf2,
        nspread=kernel_params.nspread,
        beta=float(kernel_params.beta),
        c=float(kernel_params.c),
    )
    c_real, c_imag = pl.pallas_call(
        kernel_fn,
        grid=(M_pad // BLOCK_SIZE,),
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((nf_total,), lambda i: (0,)),
            pl.BlockSpec((nf_total,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((M_pad,), jnp.float32),
            jax.ShapeDtypeStruct((M_pad,), jnp.float32),
        ],
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
    )(x_pad, y_pad, fw_real, fw_imag)
    return (c_real[:M] + 1j * c_imag[:M]).astype(fw.dtype)


# ============================================================================
# ============================================================================
# Custom JAX Primitives with transpose rules for Pallas operations
# ============================================================================
# spread transposes to interp; interp transposes to spread.
# Each primitive registers: impl, abstract_eval, transpose, jvp, batch.


def _gpu_dispatch(fn, *args):
    """Run fn(args) unless GPU is available with enough points."""
    return fn(*args)


# ===================================================================
# 2D Spread  — spread(x, y, c) → grid(nf2, nf1)
# ===================================================================
_s2 = Primitive("nufftax_spread_2d")
_s2.multiple_results = False
_s2.def_impl(
    lambda *a, nf1, nf2, kernel_params: (
        spread_2d_pallas(a[0], a[1], a[2], nf1, nf2, kernel_params)
        if _HAS_PALLAS_GPU and a[0].ndim > 0 and a[0].shape[0] >= _PALLAS_MIN_M_SPREAD
        else _s2_impl(a[0], a[1], a[2], nf1, nf2, kernel_params)
    )
)
_s2.def_abstract_eval(lambda *a, nf1, nf2, **_: ShapedArray((nf2, nf1), a[2].dtype))
_ad.primitive_transposes[_s2] = lambda ct, *a, **p: (
    jnp.zeros_like(a[0]),
    jnp.zeros_like(a[1]),
    _i2_impl(a[0], a[1], ct, p["kernel_params"]),
)
_ad.primitive_jvps[_s2] = lambda pt, tt, nf1, nf2, kernel_params: (
    _s2.impl(*pt, nf1=nf1, nf2=nf2, kernel_params=kernel_params),
    _s2.impl(
        *pt[:2], tt[2] if tt[2] is not None else jnp.zeros_like(pt[2]), nf1=nf1, nf2=nf2, kernel_params=kernel_params
    ),
)
_batching.primitive_batchers[_s2] = lambda a, ba, **p: (
    (jax.vmap(lambda d: _s2.bind(*a[:2], d, **p), 0, 0)(a[2]), ba[2])
    if ba[2] is not None
    else (_s2.bind(*a, **p), ba[2])
)


def spread_2d_primitive(x, y, c, nf1, nf2, kp):
    return _s2.bind(x, y, c, nf1=nf1, nf2=nf2, kernel_params=kp)


# ===================================================================
# 2D Interp — interp(x, y, fw) → points(M)
# ===================================================================
_i2 = Primitive("nufftax_interp_2d")
_i2.multiple_results = False
_i2.def_impl(
    lambda *a, kernel_params: (
        interp_2d_pallas(a[0], a[1], a[2], kernel_params)
        if _HAS_PALLAS_GPU and a[0].ndim > 0 and a[0].shape[0] >= _PALLAS_MIN_M_INTERP
        else _i2_impl(a[0], a[1], a[2], kernel_params)
    )
)
_i2.def_abstract_eval(lambda *a, **_: ShapedArray(a[0].shape, a[2].dtype))
_ad.primitive_transposes[_i2] = lambda ct, *a, **p: (
    jnp.zeros_like(a[0]),
    jnp.zeros_like(a[1]),
    _s2_disp(a[0], a[1], ct, a[2].shape[-1], a[2].shape[-2], p["kernel_params"]),
)
_ad.primitive_jvps[_i2] = lambda pt, tt, kernel_params: (
    _i2.impl(*pt, kernel_params=kernel_params),
    _i2.impl(*pt[:2], tt[2] if tt[2] is not None else jnp.zeros_like(pt[2]), kernel_params=kernel_params),
)
_batching.primitive_batchers[_i2] = lambda a, ba, **p: (
    (jax.vmap(lambda d: _i2.bind(*a[:2], d, **p), 0, 0)(a[2]), ba[2])
    if ba[2] is not None
    else (_i2.bind(*a, **p), ba[2])
)


def interp_2d_primitive(x, y, fw, kp):
    return _i2.bind(x, y, fw, kernel_params=kp)


# ===================================================================
# 1D Spread  — spread(x, c) → grid(nf)
# ===================================================================
_s1 = Primitive("nufftax_spread_1d")
_s1.multiple_results = False
_s1.def_impl(
    lambda *a, nf, kernel_params: (
        spread_1d_pallas(a[0], a[1], nf, kernel_params)
        if _HAS_PALLAS_GPU and a[0].ndim > 0 and a[0].shape[0] >= _PALLAS_MIN_M_SPREAD
        else _s1_impl(a[0], a[1], nf, kernel_params)
    )
)
_s1.def_abstract_eval(lambda *a, nf, **_: ShapedArray((nf,), a[1].dtype))
_ad.primitive_transposes[_s1] = lambda ct, *a, **p: (jnp.zeros_like(a[0]), _i1_impl(a[0], ct, p["kernel_params"]))
_ad.primitive_jvps[_s1] = lambda pt, tt, nf, kernel_params: (
    _s1.impl(*pt, nf=nf, kernel_params=kernel_params),
    _s1.impl(pt[0], tt[1] if tt[1] is not None else jnp.zeros_like(pt[1]), nf=nf, kernel_params=kernel_params),
)
_batching.primitive_batchers[_s1] = lambda a, ba, **p: (
    (jax.vmap(lambda d: _s1.bind(a[0], d, **p), 0, 0)(a[1]), ba[1])
    if ba[1] is not None
    else (_s1.bind(*a, **p), ba[1])
)


def spread_1d_primitive(x, c, nf, kp):
    return _s1.bind(x, c, nf=nf, kernel_params=kp)


# ===================================================================
# 1D Interp — interp(x, fw) → points(M)
# ===================================================================
_i1 = Primitive("nufftax_interp_1d")
_i1.multiple_results = False
_i1.def_impl(
    lambda *a, kernel_params: (
        interp_1d_pallas(a[0], a[1], kernel_params)
        if _HAS_PALLAS_GPU and a[0].ndim > 0 and a[0].shape[0] >= _PALLAS_MIN_M_INTERP
        else _i1_impl(a[0], a[1], kernel_params)
    )
)
_i1.def_abstract_eval(lambda *a, **_: ShapedArray(a[0].shape, a[1].dtype))
_ad.primitive_transposes[_i1] = lambda ct, *a, **p: (
    jnp.zeros_like(a[0]),
    _s1_disp(a[0], ct, nf=a[1].shape[-1], kernel_params=p["kernel_params"]),
)
_ad.primitive_jvps[_i1] = lambda pt, tt, kernel_params: (
    _i1.impl(*pt, kernel_params=kernel_params),
    _i1.impl(pt[0], tt[1] if tt[1] is not None else jnp.zeros_like(pt[1]), kernel_params=kernel_params),
)
_batching.primitive_batchers[_i1] = lambda a, ba, **p: (
    (jax.vmap(lambda d: _i1.bind(a[0], d, **p), 0, 0)(a[1]), ba[1])
    if ba[1] is not None
    else (_i1.bind(*a, **p), ba[1])
)


def interp_1d_primitive(x, fw, kp):
    return _i1.bind(x, fw, kernel_params=kp)
