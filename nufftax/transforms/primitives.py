"""JAX core primitives for the public NUFFT API.

Each public NUFFT (nufft1d1 ... nufft3d3) is a `jax.extend.core.Primitive` with:
- impl (eager)
- abstract_eval (shape/dtype)
- mlir lowering (jit) via mlir.lower_fun
- jvp rule
- transpose rule

Transpose pairs:
- Type 1 (nufftXd1) <-> Type 2 (nufftXd2), same isign (linear transpose, not gradient).
- Type 3 (nufftXd3) is self-adjoint with source/target points swapped, same isign.

The grid size n_modes for Type 3 is symmetric in (source, target) extents, so the
transpose rule reuses the forward n_modes value.
"""

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from . import nufft1 as _f1
from . import nufft2 as _f2
from . import nufft3 as _f3


def _make_batcher(prim, impl_fn, source_idx):
    """Batching rule with a fast path for the common case.

    Fast path (no overhead): only the source arg (c for Type 1/3, f for Type 2)
    is batched at axis 0. The impl handles a leading batch axis natively.

    Fallback: vmap over the impl function (pure JAX) for any other pattern —
    notably vmap over coordinates (x/y/z) with shared source.
    """

    def batcher(args, dims, **kwargs):
        batched = [i for i, d in enumerate(dims) if d is not batching.not_mapped]
        # Fast path: only the source batched at axis 0
        if batched == [source_idx] and dims[source_idx] == 0:
            return prim.bind(*args, **kwargs), 0
        # Generic fallback: vmap-trace through the impl
        in_axes = tuple(d if d is not batching.not_mapped else None for d in dims)
        out = jax.vmap(lambda *a: impl_fn(*a, **kwargs), in_axes=in_axes)(*args)
        return out, 0

    return batcher


def _zero_like(primal_out):
    return ad.Zero.from_primal_value(primal_out)


def _sum_tangents(out_tangents, primal_out):
    if not out_tangents:
        return _zero_like(primal_out)
    df = out_tangents[0]
    for t in out_tangents[1:]:
        df = df + t
    return df


def _register(prim, impl_fn, aval_fn, jvp_fn, transpose_fn, source_idx):
    prim.def_impl(impl_fn)
    prim.def_abstract_eval(aval_fn)
    mlir.register_lowering(prim, mlir.lower_fun(impl_fn, multiple_results=False))
    ad.primitive_jvps[prim] = jvp_fn
    ad.primitive_transposes[prim] = transpose_fn
    batching.primitive_batchers[prim] = _make_batcher(prim, impl_fn, source_idx)


# ============================================================================
# 1D Type 1
# ============================================================================

nufft1d1_p = Primitive("nufft1d1")


def _impl_1d1(x, c, *, n_modes, eps, isign):
    return _f1.nufft1d1(x, c, n_modes, eps, isign)


def _aval_1d1(x, c, *, n_modes, eps, isign):
    return jax.core.ShapedArray(c.shape[:-1] + (n_modes,), c.dtype)


def _jvp_1d1(primals, tangents, *, n_modes, eps, isign):
    x, c = primals
    dx, dc = tangents
    f = nufft1d1_p.bind(x, c, n_modes=n_modes, eps=eps, isign=isign)

    out = []
    if not isinstance(dc, ad.Zero):
        out.append(nufft1d1_p.bind(x, dc, n_modes=n_modes, eps=eps, isign=isign))
    if not isinstance(dx, ad.Zero):
        k = jnp.arange(-(n_modes // 2), (n_modes + 1) // 2)
        t = nufft1d1_p.bind(x, c * dx, n_modes=n_modes, eps=eps, isign=isign)
        out.append(1j * isign * k * t)
    return f, _sum_tangents(out, f)


def _transpose_1d1(cot, x, c, *, n_modes, eps, isign):
    assert ad.is_undefined_primal(c)
    cot_c = nufft1d2_p.bind(x, cot, eps=eps, isign=isign)
    return (None, cot_c)


# ============================================================================
# 1D Type 2
# ============================================================================

nufft1d2_p = Primitive("nufft1d2")


def _impl_1d2(x, f, *, eps, isign):
    return _f2.nufft1d2(x, f, eps, isign)


def _aval_1d2(x, f, *, eps, isign):
    return jax.core.ShapedArray(f.shape[:-1] + (x.shape[0],), f.dtype)


def _jvp_1d2(primals, tangents, *, eps, isign):
    x, f = primals
    dx, df = tangents
    n_modes = f.shape[-1]
    c = nufft1d2_p.bind(x, f, eps=eps, isign=isign)

    out = []
    if not isinstance(df, ad.Zero):
        out.append(nufft1d2_p.bind(x, df, eps=eps, isign=isign))
    if not isinstance(dx, ad.Zero):
        k = jnp.arange(-(n_modes // 2), (n_modes + 1) // 2)
        kf = k * f
        t = nufft1d2_p.bind(x, kf, eps=eps, isign=isign)
        out.append(1j * isign * dx * t)
    return c, _sum_tangents(out, c)


def _transpose_1d2(cot, x, f, *, eps, isign):
    assert ad.is_undefined_primal(f)
    n_modes = f.aval.shape[-1]
    cot_f = nufft1d1_p.bind(x, cot, n_modes=n_modes, eps=eps, isign=isign)
    return (None, cot_f)


# ============================================================================
# 2D Type 1
# ============================================================================

nufft2d1_p = Primitive("nufft2d1")


def _impl_2d1(x, y, c, *, n_modes, eps, isign):
    return _f1.nufft2d1(x, y, c, n_modes, eps, isign)


def _aval_2d1(x, y, c, *, n_modes, eps, isign):
    n1, n2 = n_modes
    return jax.core.ShapedArray(c.shape[:-1] + (n2, n1), c.dtype)


def _jvp_2d1(primals, tangents, *, n_modes, eps, isign):
    x, y, c = primals
    dx, dy, dc = tangents
    n1, n2 = n_modes
    f = nufft2d1_p.bind(x, y, c, n_modes=n_modes, eps=eps, isign=isign)

    out = []
    if not isinstance(dc, ad.Zero):
        out.append(nufft2d1_p.bind(x, y, dc, n_modes=n_modes, eps=eps, isign=isign))
    if not isinstance(dx, ad.Zero):
        k1 = jnp.arange(-(n1 // 2), (n1 + 1) // 2)
        t = nufft2d1_p.bind(x, y, c * dx, n_modes=n_modes, eps=eps, isign=isign)
        out.append(1j * isign * k1[None, :] * t)
    if not isinstance(dy, ad.Zero):
        k2 = jnp.arange(-(n2 // 2), (n2 + 1) // 2)
        t = nufft2d1_p.bind(x, y, c * dy, n_modes=n_modes, eps=eps, isign=isign)
        out.append(1j * isign * k2[:, None] * t)
    return f, _sum_tangents(out, f)


def _transpose_2d1(cot, x, y, c, *, n_modes, eps, isign):
    assert ad.is_undefined_primal(c)
    cot_c = nufft2d2_p.bind(x, y, cot, eps=eps, isign=isign)
    return (None, None, cot_c)


# ============================================================================
# 2D Type 2
# ============================================================================

nufft2d2_p = Primitive("nufft2d2")


def _impl_2d2(x, y, f, *, eps, isign):
    return _f2.nufft2d2(x, y, f, eps, isign)


def _aval_2d2(x, y, f, *, eps, isign):
    return jax.core.ShapedArray(f.shape[:-2] + (x.shape[0],), f.dtype)


def _jvp_2d2(primals, tangents, *, eps, isign):
    x, y, f = primals
    dx, dy, df = tangents
    n2, n1 = f.shape[-2:]
    c = nufft2d2_p.bind(x, y, f, eps=eps, isign=isign)

    out = []
    if not isinstance(df, ad.Zero):
        out.append(nufft2d2_p.bind(x, y, df, eps=eps, isign=isign))
    if not isinstance(dx, ad.Zero):
        k1 = jnp.arange(-(n1 // 2), (n1 + 1) // 2)
        k1_f = f * k1[None, :]
        t = nufft2d2_p.bind(x, y, k1_f, eps=eps, isign=isign)
        out.append(1j * isign * dx * t)
    if not isinstance(dy, ad.Zero):
        k2 = jnp.arange(-(n2 // 2), (n2 + 1) // 2)
        k2_f = f * k2[:, None]
        t = nufft2d2_p.bind(x, y, k2_f, eps=eps, isign=isign)
        out.append(1j * isign * dy * t)
    return c, _sum_tangents(out, c)


def _transpose_2d2(cot, x, y, f, *, eps, isign):
    assert ad.is_undefined_primal(f)
    n2, n1 = f.aval.shape[-2:]
    cot_f = nufft2d1_p.bind(x, y, cot, n_modes=(n1, n2), eps=eps, isign=isign)
    return (None, None, cot_f)


# ============================================================================
# 3D Type 1
# ============================================================================

nufft3d1_p = Primitive("nufft3d1")


def _impl_3d1(x, y, z, c, *, n_modes, eps, isign):
    return _f1.nufft3d1(x, y, z, c, n_modes, eps, isign)


def _aval_3d1(x, y, z, c, *, n_modes, eps, isign):
    n1, n2, n3 = n_modes
    return jax.core.ShapedArray(c.shape[:-1] + (n3, n2, n1), c.dtype)


def _jvp_3d1(primals, tangents, *, n_modes, eps, isign):
    x, y, z, c = primals
    dx, dy, dz, dc = tangents
    n1, n2, n3 = n_modes
    f = nufft3d1_p.bind(x, y, z, c, n_modes=n_modes, eps=eps, isign=isign)

    out = []
    if not isinstance(dc, ad.Zero):
        out.append(nufft3d1_p.bind(x, y, z, dc, n_modes=n_modes, eps=eps, isign=isign))
    if not isinstance(dx, ad.Zero):
        k1 = jnp.arange(-(n1 // 2), (n1 + 1) // 2)
        t = nufft3d1_p.bind(x, y, z, c * dx, n_modes=n_modes, eps=eps, isign=isign)
        out.append(1j * isign * k1[None, None, :] * t)
    if not isinstance(dy, ad.Zero):
        k2 = jnp.arange(-(n2 // 2), (n2 + 1) // 2)
        t = nufft3d1_p.bind(x, y, z, c * dy, n_modes=n_modes, eps=eps, isign=isign)
        out.append(1j * isign * k2[None, :, None] * t)
    if not isinstance(dz, ad.Zero):
        k3 = jnp.arange(-(n3 // 2), (n3 + 1) // 2)
        t = nufft3d1_p.bind(x, y, z, c * dz, n_modes=n_modes, eps=eps, isign=isign)
        out.append(1j * isign * k3[:, None, None] * t)
    return f, _sum_tangents(out, f)


def _transpose_3d1(cot, x, y, z, c, *, n_modes, eps, isign):
    assert ad.is_undefined_primal(c)
    cot_c = nufft3d2_p.bind(x, y, z, cot, eps=eps, isign=isign)
    return (None, None, None, cot_c)


# ============================================================================
# 3D Type 2
# ============================================================================

nufft3d2_p = Primitive("nufft3d2")


def _impl_3d2(x, y, z, f, *, eps, isign):
    return _f2.nufft3d2(x, y, z, f, eps, isign)


def _aval_3d2(x, y, z, f, *, eps, isign):
    return jax.core.ShapedArray(f.shape[:-3] + (x.shape[0],), f.dtype)


def _jvp_3d2(primals, tangents, *, eps, isign):
    x, y, z, f = primals
    dx, dy, dz, df = tangents
    n3, n2, n1 = f.shape[-3:]
    c = nufft3d2_p.bind(x, y, z, f, eps=eps, isign=isign)

    out = []
    if not isinstance(df, ad.Zero):
        out.append(nufft3d2_p.bind(x, y, z, df, eps=eps, isign=isign))
    if not isinstance(dx, ad.Zero):
        k1 = jnp.arange(-(n1 // 2), (n1 + 1) // 2)
        k1_f = f * k1[None, None, :]
        t = nufft3d2_p.bind(x, y, z, k1_f, eps=eps, isign=isign)
        out.append(1j * isign * dx * t)
    if not isinstance(dy, ad.Zero):
        k2 = jnp.arange(-(n2 // 2), (n2 + 1) // 2)
        k2_f = f * k2[None, :, None]
        t = nufft3d2_p.bind(x, y, z, k2_f, eps=eps, isign=isign)
        out.append(1j * isign * dy * t)
    if not isinstance(dz, ad.Zero):
        k3 = jnp.arange(-(n3 // 2), (n3 + 1) // 2)
        k3_f = f * k3[:, None, None]
        t = nufft3d2_p.bind(x, y, z, k3_f, eps=eps, isign=isign)
        out.append(1j * isign * dz * t)
    return c, _sum_tangents(out, c)


def _transpose_3d2(cot, x, y, z, f, *, eps, isign):
    assert ad.is_undefined_primal(f)
    n3, n2, n1 = f.aval.shape[-3:]
    cot_f = nufft3d1_p.bind(x, y, z, cot, n_modes=(n1, n2, n3), eps=eps, isign=isign)
    return (None, None, None, cot_f)


# ============================================================================
# 1D Type 3
# ============================================================================

nufft1d3_p = Primitive("nufft1d3")


def _impl_1d3(x, c, s, *, n_modes, eps, isign, upsampfac):
    return _f3.nufft1d3(x, c, s, n_modes, eps, isign, upsampfac)


def _aval_1d3(x, c, s, *, n_modes, eps, isign, upsampfac):
    return jax.core.ShapedArray(c.shape[:-1] + (s.shape[0],), c.dtype)


def _jvp_1d3(primals, tangents, *, n_modes, eps, isign, upsampfac):
    x, c, s = primals
    dx, dc, ds = tangents
    kw = {"n_modes": n_modes, "eps": eps, "isign": isign, "upsampfac": upsampfac}
    f = nufft1d3_p.bind(x, c, s, **kw)

    out = []
    if not isinstance(dc, ad.Zero):
        out.append(nufft1d3_p.bind(x, dc, s, **kw))
    if not isinstance(dx, ad.Zero):
        t = nufft1d3_p.bind(x, c * dx, s, **kw)
        out.append(1j * isign * s * t)
    if not isinstance(ds, ad.Zero):
        t = nufft1d3_p.bind(x, x * c, s, **kw)
        out.append(1j * isign * ds * t)
    return f, _sum_tangents(out, f)


def _transpose_1d3(cot, x, c, s, *, n_modes, eps, isign, upsampfac):
    assert ad.is_undefined_primal(c)
    # n_modes is symmetric in (source, target) extents -> reuse forward value.
    cot_c = nufft1d3_p.bind(s, cot, x, n_modes=n_modes, eps=eps, isign=isign, upsampfac=upsampfac)
    return (None, cot_c, None)


# ============================================================================
# 2D Type 3
# ============================================================================

nufft2d3_p = Primitive("nufft2d3")


def _impl_2d3(x, y, c, s, t, *, n_modes, eps, isign, upsampfac):
    return _f3.nufft2d3(x, y, c, s, t, n_modes, eps, isign, upsampfac)


def _aval_2d3(x, y, c, s, t, *, n_modes, eps, isign, upsampfac):
    return jax.core.ShapedArray(c.shape[:-1] + (s.shape[0],), c.dtype)


def _jvp_2d3(primals, tangents, *, n_modes, eps, isign, upsampfac):
    x, y, c, s, t = primals
    dx, dy, dc, ds, dt = tangents
    kw = {"n_modes": n_modes, "eps": eps, "isign": isign, "upsampfac": upsampfac}
    f = nufft2d3_p.bind(x, y, c, s, t, **kw)

    out = []
    if not isinstance(dc, ad.Zero):
        out.append(nufft2d3_p.bind(x, y, dc, s, t, **kw))
    if not isinstance(dx, ad.Zero):
        w = nufft2d3_p.bind(x, y, c * dx, s, t, **kw)
        out.append(1j * isign * s * w)
    if not isinstance(dy, ad.Zero):
        w = nufft2d3_p.bind(x, y, c * dy, s, t, **kw)
        out.append(1j * isign * t * w)
    if not isinstance(ds, ad.Zero):
        w = nufft2d3_p.bind(x, y, x * c, s, t, **kw)
        out.append(1j * isign * ds * w)
    if not isinstance(dt, ad.Zero):
        w = nufft2d3_p.bind(x, y, y * c, s, t, **kw)
        out.append(1j * isign * dt * w)
    return f, _sum_tangents(out, f)


def _transpose_2d3(cot, x, y, c, s, t, *, n_modes, eps, isign, upsampfac):
    assert ad.is_undefined_primal(c)
    cot_c = nufft2d3_p.bind(s, t, cot, x, y, n_modes=n_modes, eps=eps, isign=isign, upsampfac=upsampfac)
    return (None, None, cot_c, None, None)


# ============================================================================
# 3D Type 3
# ============================================================================

nufft3d3_p = Primitive("nufft3d3")


def _impl_3d3(x, y, z, c, s, t, u, *, n_modes, eps, isign, upsampfac):
    return _f3.nufft3d3(x, y, z, c, s, t, u, n_modes, eps, isign, upsampfac)


def _aval_3d3(x, y, z, c, s, t, u, *, n_modes, eps, isign, upsampfac):
    return jax.core.ShapedArray(c.shape[:-1] + (s.shape[0],), c.dtype)


def _jvp_3d3(primals, tangents, *, n_modes, eps, isign, upsampfac):
    x, y, z, c, s, t, u = primals
    dx, dy, dz, dc, ds, dt, du = tangents
    kw = {"n_modes": n_modes, "eps": eps, "isign": isign, "upsampfac": upsampfac}
    f = nufft3d3_p.bind(x, y, z, c, s, t, u, **kw)

    out = []
    if not isinstance(dc, ad.Zero):
        out.append(nufft3d3_p.bind(x, y, z, dc, s, t, u, **kw))
    if not isinstance(dx, ad.Zero):
        w = nufft3d3_p.bind(x, y, z, c * dx, s, t, u, **kw)
        out.append(1j * isign * s * w)
    if not isinstance(dy, ad.Zero):
        w = nufft3d3_p.bind(x, y, z, c * dy, s, t, u, **kw)
        out.append(1j * isign * t * w)
    if not isinstance(dz, ad.Zero):
        w = nufft3d3_p.bind(x, y, z, c * dz, s, t, u, **kw)
        out.append(1j * isign * u * w)
    if not isinstance(ds, ad.Zero):
        w = nufft3d3_p.bind(x, y, z, x * c, s, t, u, **kw)
        out.append(1j * isign * ds * w)
    if not isinstance(dt, ad.Zero):
        w = nufft3d3_p.bind(x, y, z, y * c, s, t, u, **kw)
        out.append(1j * isign * dt * w)
    if not isinstance(du, ad.Zero):
        w = nufft3d3_p.bind(x, y, z, z * c, s, t, u, **kw)
        out.append(1j * isign * du * w)
    return f, _sum_tangents(out, f)


def _transpose_3d3(cot, x, y, z, c, s, t, u, *, n_modes, eps, isign, upsampfac):
    assert ad.is_undefined_primal(c)
    cot_c = nufft3d3_p.bind(s, t, u, cot, x, y, z, n_modes=n_modes, eps=eps, isign=isign, upsampfac=upsampfac)
    return (None, None, None, cot_c, None, None, None)


# ============================================================================
# Register everything
# ============================================================================

_register(nufft1d1_p, _impl_1d1, _aval_1d1, _jvp_1d1, _transpose_1d1, source_idx=1)
_register(nufft1d2_p, _impl_1d2, _aval_1d2, _jvp_1d2, _transpose_1d2, source_idx=1)
_register(nufft2d1_p, _impl_2d1, _aval_2d1, _jvp_2d1, _transpose_2d1, source_idx=2)
_register(nufft2d2_p, _impl_2d2, _aval_2d2, _jvp_2d2, _transpose_2d2, source_idx=2)
_register(nufft3d1_p, _impl_3d1, _aval_3d1, _jvp_3d1, _transpose_3d1, source_idx=3)
_register(nufft3d2_p, _impl_3d2, _aval_3d2, _jvp_3d2, _transpose_3d2, source_idx=3)
_register(nufft1d3_p, _impl_1d3, _aval_1d3, _jvp_1d3, _transpose_1d3, source_idx=1)
_register(nufft2d3_p, _impl_2d3, _aval_2d3, _jvp_2d3, _transpose_2d3, source_idx=2)
_register(nufft3d3_p, _impl_3d3, _aval_3d3, _jvp_3d3, _transpose_3d3, source_idx=3)
