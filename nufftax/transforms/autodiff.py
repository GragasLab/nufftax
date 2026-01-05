"""
Custom autodiff rules for JAX-FINUFFT.

This module provides custom VJP (reverse-mode) and JVP (forward-mode)
implementations for NUFFT operations, enabling efficient gradient computation.

Key insight: Type 1 and Type 2 NUFFTs are adjoints of each other (with sign flip).
This means:
- Gradient of nufft1 w.r.t. `c` is nufft2 applied to the upstream gradient
- Gradient of nufft2 w.r.t. `f` is nufft1 applied to the upstream gradient

Mathematical relationships:
- <nufft1(x, c), f> = <c, nufft2(x, f)>  (adjoint property)

Reference: .claude/agents/autodiff-dev.md
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

# =============================================================================
# Internal implementation placeholders
# These should be imported from the actual transform modules when available
# =============================================================================


def _nufft1d1_impl(x: Array, c: Array, n_modes: int, eps: float = 1e-6, isign: int = 1) -> Array:
    """
    Implementation of 1D Type 1 NUFFT (nonuniform to uniform).

    f[k] = sum_j c[j] * exp(i * isign * k * x[j])
    """
    from .nufft1 import nufft1d1 as nufft1d1_fast

    return nufft1d1_fast(x, c, n_modes, eps, isign)


def _nufft1d2_impl(x: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """
    Implementation of 1D Type 2 NUFFT (uniform to nonuniform).

    c[j] = sum_k f[k] * exp(i * isign * k * x[j])
    """
    from .nufft2 import nufft1d2 as nufft1d2_fast

    return nufft1d2_fast(x, f, eps, isign)


# =============================================================================
# 1D Type 1 NUFFT with custom VJP
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def nufft1d1(x: Array, c: Array, n_modes: int, eps: float = 1e-6, isign: int = 1) -> Array:
    """
    1D Type 1 NUFFT with custom gradients.

    Computes:
        f[k] = sum_j c[j] * exp(i * isign * k * x[j])

    for k in {-n_modes//2, ..., n_modes//2 - 1}.

    Args:
        x: Nonuniform point locations in [-pi, pi], shape (M,)
        c: Complex strengths at nonuniform points, shape (M,)
        n_modes: Number of Fourier modes to compute
        eps: Requested precision (tolerance)
        isign: Sign of exponent (+1 or -1)

    Returns:
        f: Fourier coefficients, shape (n_modes,)
    """
    return _nufft1d1_impl(x, c, n_modes, eps, isign)


def _nufft1d1_fwd(
    x: Array,
    c: Array,
    n_modes: int,
    eps: float,
    isign: int,
) -> tuple[Array, tuple]:
    """Forward pass for nufft1d1 VJP.

    Note: With nondiff_argnums=(2, 3, 4), the signature is:
    (diff_args..., nondiff_args...) = (x, c, n_modes, eps, isign)
    """
    f = _nufft1d1_impl(x, c, n_modes, eps, isign)
    # Save values needed for backward pass (only diff args)
    return f, (x, c)


def _nufft1d1_bwd(
    n_modes: int, eps: float, isign: int, res: tuple, g: Array
) -> tuple[Array | None, Array | None]:
    """
    Backward pass for nufft1d1 VJP.

    Gradient relationships:
    - grad w.r.t. c: Type 2 with opposite sign
    - grad w.r.t. x: derivative of exponential kernel

    Note on JAX's complex gradient convention:
    JAX passes the cotangent g = dL/d(conj f) for complex outputs.
    For L = sum|f|^2, this gives g = f.
    """
    x, c = res

    # Conjugate g for correct gradient through holomorphic function
    g_conj = jnp.conj(g)

    # Gradient w.r.t. c: adjoint is Type 2 with opposite isign
    dc = _nufft1d2_impl(x, g_conj, eps, -isign)

    # Gradient w.r.t. x:
    # f[k] = sum_j c[j] * exp(i * isign * k * x[j])
    # df/dx[j] = i * isign * k * c[j] * exp(i * isign * k * x[j])
    #
    # For real loss L:
    # dL/dx[j] = 2 * isign * Im(conj(c[j]) * sum_k k * g[k] * exp(-i * isign * k * x[j]))
    #          = 2 * Re(conj(c[j]) * (-i * isign) * Type2(k * g, -isign)[j])

    k = jnp.arange(-(n_modes // 2), (n_modes + 1) // 2)
    kg = k * g_conj  # Weight by k

    # Compute weighted transform with -isign
    dx_complex = _nufft1d2_impl(x, kg, eps, -isign)

    # Apply chain rule: use -1j and factor of 1 (not 2)
    dx = jnp.real(jnp.conj(c) * (-1j) * isign * dx_complex)

    return (dx, dc)


nufft1d1.defvjp(_nufft1d1_fwd, _nufft1d1_bwd)


# =============================================================================
# 1D Type 2 NUFFT with custom VJP
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def nufft1d2(x: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """
    1D Type 2 NUFFT with custom gradients.

    Computes:
        c[j] = sum_k f[k] * exp(i * isign * k * x[j])

    Args:
        x: Nonuniform point locations in [-pi, pi], shape (M,)
        f: Fourier coefficients, shape (n_modes,)
        eps: Requested precision (tolerance)
        isign: Sign of exponent (+1 or -1)

    Returns:
        c: Complex values at nonuniform points, shape (M,)
    """
    return _nufft1d2_impl(x, f, eps, isign)


def _nufft1d2_fwd(
    x: Array,
    f: Array,
    eps: float,
    isign: int,
) -> tuple[Array, tuple]:
    """Forward pass for nufft1d2 VJP.

    Note: With nondiff_argnums=(2, 3), the signature is:
    (diff_args..., nondiff_args...) = (x, f, eps, isign)
    """
    c = _nufft1d2_impl(x, f, eps, isign)
    return c, (x, f)


def _nufft1d2_bwd(
    eps: float, isign: int, res: tuple, g: Array
) -> tuple[Array | None, Array | None]:
    """
    Backward pass for nufft1d2 VJP.

    Gradient relationships:
    - grad w.r.t. f: Type 1 with opposite sign
    - grad w.r.t. x: derivative of exponential kernel

    Note on JAX's complex gradient convention:
    JAX passes the cotangent g which for L = sum|c|^2 equals 2*conj(c).
    The gradient w.r.t. x is: Re(conj(g) * (-i) * isign * Type2(k*conj(f), -isign))
    """
    x, f = res
    n_modes = f.shape[-1]

    # Conjugate g for correct gradient through holomorphic function
    g_conj = jnp.conj(g)

    # Gradient w.r.t. f: adjoint is Type 1 with opposite isign
    df = _nufft1d1_impl(x, g_conj, n_modes, eps, -isign)

    # Gradient w.r.t. x:
    # dc[j]/dx[j] = i * isign * Type2(k*f, isign)[j]
    # conj(dc[j]/dx[j]) = -i * isign * Type2(k*conj(f), -isign)[j]
    # dL/dx = Re(conj(g) * conj(dc/dx)) = Re(g_conj * (-i*isign) * Type2(k*conj(f), -isign))

    k = jnp.arange(-(n_modes // 2), (n_modes + 1) // 2)
    k_conj_f = k * jnp.conj(f)  # Weight by k and conjugate f

    # Compute weighted transform with -isign
    dx_complex = _nufft1d2_impl(x, k_conj_f, eps, -isign)

    # Apply chain rule
    dx = jnp.real(g_conj * (-1j) * isign * dx_complex)

    return (dx, df)


nufft1d2.defvjp(_nufft1d2_fwd, _nufft1d2_bwd)


# =============================================================================
# 1D Type 1 NUFFT with custom JVP (forward mode)
# =============================================================================


@jax.custom_jvp
def nufft1d1_jvp(x: Array, c: Array, n_modes: int, eps: float = 1e-6, isign: int = 1) -> Array:
    """
    1D Type 1 NUFFT with forward-mode AD support.

    Same as nufft1d1 but with JVP instead of VJP.
    """
    return _nufft1d1_impl(x, c, n_modes, eps, isign)


@nufft1d1_jvp.defjvp
def _nufft1d1_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """
    Forward-mode differentiation for nufft1d1.

    Computes both primal output and tangent output in one pass.
    """
    x, c, n_modes, eps, isign = primals
    dx, dc, _, _, _ = tangents

    # Primal output
    f = _nufft1d1_impl(x, c, n_modes, eps, isign)

    # Tangent output
    # df = df/dc * dc + df/dx * dx

    # Contribution from dc: standard Type 1 transform of dc
    df_from_c = _nufft1d1_impl(x, dc, n_modes, eps, isign) if dc is not None else jnp.zeros_like(f)

    # Contribution from dx:
    # f[k] = sum_j c[j] * exp(i * isign * k * x[j])
    # df/dx[j] = i * isign * k * c[j] * exp(i * isign * k * x[j])
    # df = sum_j dx[j] * df/dx[j]
    #    = i * isign * k * sum_j (c[j] * dx[j]) * exp(i * isign * k * x[j])
    #    = i * isign * k * nufft1(x, c * dx)
    if dx is not None:
        k = jnp.arange(-(n_modes // 2), (n_modes + 1) // 2)
        # Transform of c * dx
        weighted_transform = _nufft1d1_impl(x, c * dx, n_modes, eps, isign)
        df_from_x = 1j * isign * k * weighted_transform
    else:
        df_from_x = jnp.zeros_like(f)

    df = df_from_c + df_from_x

    return f, df


# =============================================================================
# 1D Type 2 NUFFT with custom JVP (forward mode)
# =============================================================================


@jax.custom_jvp
def nufft1d2_jvp(x: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """
    1D Type 2 NUFFT with forward-mode AD support.

    Same as nufft1d2 but with JVP instead of VJP.
    """
    return _nufft1d2_impl(x, f, eps, isign)


@nufft1d2_jvp.defjvp
def _nufft1d2_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """
    Forward-mode differentiation for nufft1d2.
    """
    x, f, eps, isign = primals
    dx, df, _, _ = tangents
    n_modes = f.shape[-1]

    # Primal output
    c = _nufft1d2_impl(x, f, eps, isign)

    # Tangent output

    # Contribution from df: standard Type 2 transform of df
    dc_from_f = _nufft1d2_impl(x, df, eps, isign) if df is not None else jnp.zeros_like(c)

    # Contribution from dx:
    # c[j] = sum_k f[k] * exp(i * isign * k * x[j])
    # dc[j]/dx[j] = i * isign * sum_k k * f[k] * exp(i * isign * k * x[j])
    # dc[j] = dx[j] * dc[j]/dx[j]
    if dx is not None:
        k = jnp.arange(-(n_modes // 2), (n_modes + 1) // 2)
        kf = k * f
        # Type 2 transform of k-weighted coefficients
        weighted_transform = _nufft1d2_impl(x, kf, eps, isign)
        dc_from_x = 1j * isign * dx * weighted_transform
    else:
        dc_from_x = jnp.zeros_like(c)

    dc = dc_from_f + dc_from_x

    return c, dc


# =============================================================================
# 2D Type 1 NUFFT with custom JVP (forward mode)
# =============================================================================


@jax.custom_jvp
def nufft2d1_jvp(
    x: Array, y: Array, c: Array, n_modes: tuple[int, int], eps: float = 1e-6, isign: int = 1
) -> Array:
    """
    2D Type 1 NUFFT with forward-mode AD support.

    Same as nufft2d1 but with JVP instead of VJP.
    """
    return _nufft2d1_impl(x, y, c, n_modes, eps, isign)


@nufft2d1_jvp.defjvp
def _nufft2d1_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """
    Forward-mode differentiation for nufft2d1.
    """
    x, y, c, n_modes, eps, isign = primals
    dx, dy, dc, _, _, _ = tangents
    n1, n2 = n_modes

    # Primal output
    f = _nufft2d1_impl(x, y, c, n_modes, eps, isign)

    # Tangent output: df = df/dc * dc + df/dx * dx + df/dy * dy

    # Contribution from dc: standard Type 1 transform of dc
    df_from_c = (
        _nufft2d1_impl(x, y, dc, n_modes, eps, isign) if dc is not None else jnp.zeros_like(f)
    )

    # Contribution from dx:
    # f[k1, k2] = sum_j c[j] * exp(i * isign * (k1*x[j] + k2*y[j]))
    # df/dx[j] = i * isign * k1 * c[j] * exp(...)
    # df = i * isign * k1 * nufft2d1(x, y, c * dx)
    if dx is not None:
        k1 = jnp.arange(-(n1 // 2), (n1 + 1) // 2)
        weighted_transform = _nufft2d1_impl(x, y, c * dx, n_modes, eps, isign)
        df_from_x = 1j * isign * k1[None, :] * weighted_transform  # k1 on axis 1
    else:
        df_from_x = jnp.zeros_like(f)

    # Contribution from dy:
    # df/dy[j] = i * isign * k2 * c[j] * exp(...)
    if dy is not None:
        k2 = jnp.arange(-(n2 // 2), (n2 + 1) // 2)
        weighted_transform = _nufft2d1_impl(x, y, c * dy, n_modes, eps, isign)
        df_from_y = 1j * isign * k2[:, None] * weighted_transform  # k2 on axis 0
    else:
        df_from_y = jnp.zeros_like(f)

    df = df_from_c + df_from_x + df_from_y

    return f, df


# =============================================================================
# 2D Type 2 NUFFT with custom JVP (forward mode)
# =============================================================================


@jax.custom_jvp
def nufft2d2_jvp(x: Array, y: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """
    2D Type 2 NUFFT with forward-mode AD support.

    Same as nufft2d2 but with JVP instead of VJP.
    """
    return _nufft2d2_impl(x, y, f, eps, isign)


@nufft2d2_jvp.defjvp
def _nufft2d2_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """
    Forward-mode differentiation for nufft2d2.
    """
    x, y, f, eps, isign = primals
    dx, dy, df, _, _ = tangents
    n2, n1 = f.shape  # f has shape (n2, n1)

    # Primal output
    c = _nufft2d2_impl(x, y, f, eps, isign)

    # Tangent output: dc = dc/df * df + dc/dx * dx + dc/dy * dy

    # Contribution from df: standard Type 2 transform of df
    dc_from_f = _nufft2d2_impl(x, y, df, eps, isign) if df is not None else jnp.zeros_like(c)

    # Contribution from dx:
    # c[j] = sum_{k1,k2} f[k1,k2] * exp(i * isign * (k1*x[j] + k2*y[j]))
    # dc[j]/dx[j] = i * isign * sum_{k1,k2} k1 * f[k1,k2] * exp(...)
    if dx is not None:
        k1 = jnp.arange(-(n1 // 2), (n1 + 1) // 2)
        k1_f = f * k1[None, :]  # k1 on axis 1
        weighted_transform = _nufft2d2_impl(x, y, k1_f, eps, isign)
        dc_from_x = 1j * isign * dx * weighted_transform
    else:
        dc_from_x = jnp.zeros_like(c)

    # Contribution from dy:
    # dc[j]/dy[j] = i * isign * sum_{k1,k2} k2 * f[k1,k2] * exp(...)
    if dy is not None:
        k2 = jnp.arange(-(n2 // 2), (n2 + 1) // 2)
        k2_f = f * k2[:, None]  # k2 on axis 0
        weighted_transform = _nufft2d2_impl(x, y, k2_f, eps, isign)
        dc_from_y = 1j * isign * dy * weighted_transform
    else:
        dc_from_y = jnp.zeros_like(c)

    dc = dc_from_f + dc_from_x + dc_from_y

    return c, dc


# =============================================================================
# 3D Type 1 NUFFT with custom JVP (forward mode)
# =============================================================================


@jax.custom_jvp
def nufft3d1_jvp(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
) -> Array:
    """
    3D Type 1 NUFFT with forward-mode AD support.

    Same as nufft3d1 but with JVP instead of VJP.
    """
    return _nufft3d1_impl(x, y, z, c, n_modes, eps, isign)


@nufft3d1_jvp.defjvp
def _nufft3d1_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """
    Forward-mode differentiation for nufft3d1.
    """
    x, y, z, c, n_modes, eps, isign = primals
    dx, dy, dz, dc, _, _, _ = tangents
    n1, n2, n3 = n_modes

    # Primal output
    f = _nufft3d1_impl(x, y, z, c, n_modes, eps, isign)

    # Tangent output: df = df/dc * dc + df/dx * dx + df/dy * dy + df/dz * dz

    # Contribution from dc
    df_from_c = (
        _nufft3d1_impl(x, y, z, dc, n_modes, eps, isign) if dc is not None else jnp.zeros_like(f)
    )

    # Contribution from dx: k1 on axis 2
    if dx is not None:
        k1 = jnp.arange(-(n1 // 2), (n1 + 1) // 2)
        weighted_transform = _nufft3d1_impl(x, y, z, c * dx, n_modes, eps, isign)
        df_from_x = 1j * isign * k1[None, None, :] * weighted_transform
    else:
        df_from_x = jnp.zeros_like(f)

    # Contribution from dy: k2 on axis 1
    if dy is not None:
        k2 = jnp.arange(-(n2 // 2), (n2 + 1) // 2)
        weighted_transform = _nufft3d1_impl(x, y, z, c * dy, n_modes, eps, isign)
        df_from_y = 1j * isign * k2[None, :, None] * weighted_transform
    else:
        df_from_y = jnp.zeros_like(f)

    # Contribution from dz: k3 on axis 0
    if dz is not None:
        k3 = jnp.arange(-(n3 // 2), (n3 + 1) // 2)
        weighted_transform = _nufft3d1_impl(x, y, z, c * dz, n_modes, eps, isign)
        df_from_z = 1j * isign * k3[:, None, None] * weighted_transform
    else:
        df_from_z = jnp.zeros_like(f)

    df = df_from_c + df_from_x + df_from_y + df_from_z

    return f, df


# =============================================================================
# 3D Type 2 NUFFT with custom JVP (forward mode)
# =============================================================================


@jax.custom_jvp
def nufft3d2_jvp(
    x: Array, y: Array, z: Array, f: Array, eps: float = 1e-6, isign: int = -1
) -> Array:
    """
    3D Type 2 NUFFT with forward-mode AD support.

    Same as nufft3d2 but with JVP instead of VJP.
    """
    return _nufft3d2_impl(x, y, z, f, eps, isign)


@nufft3d2_jvp.defjvp
def _nufft3d2_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """
    Forward-mode differentiation for nufft3d2.
    """
    x, y, z, f, eps, isign = primals
    dx, dy, dz, df, _, _ = tangents
    n3, n2, n1 = f.shape  # f has shape (n3, n2, n1)

    # Primal output
    c = _nufft3d2_impl(x, y, z, f, eps, isign)

    # Tangent output: dc = dc/df * df + dc/dx * dx + dc/dy * dy + dc/dz * dz

    # Contribution from df
    dc_from_f = _nufft3d2_impl(x, y, z, df, eps, isign) if df is not None else jnp.zeros_like(c)

    # Contribution from dx: k1 on axis 2
    if dx is not None:
        k1 = jnp.arange(-(n1 // 2), (n1 + 1) // 2)
        k1_f = f * k1[None, None, :]
        weighted_transform = _nufft3d2_impl(x, y, z, k1_f, eps, isign)
        dc_from_x = 1j * isign * dx * weighted_transform
    else:
        dc_from_x = jnp.zeros_like(c)

    # Contribution from dy: k2 on axis 1
    if dy is not None:
        k2 = jnp.arange(-(n2 // 2), (n2 + 1) // 2)
        k2_f = f * k2[None, :, None]
        weighted_transform = _nufft3d2_impl(x, y, z, k2_f, eps, isign)
        dc_from_y = 1j * isign * dy * weighted_transform
    else:
        dc_from_y = jnp.zeros_like(c)

    # Contribution from dz: k3 on axis 0
    if dz is not None:
        k3 = jnp.arange(-(n3 // 2), (n3 + 1) // 2)
        k3_f = f * k3[:, None, None]
        weighted_transform = _nufft3d2_impl(x, y, z, k3_f, eps, isign)
        dc_from_z = 1j * isign * dz * weighted_transform
    else:
        dc_from_z = jnp.zeros_like(c)

    dc = dc_from_f + dc_from_x + dc_from_y + dc_from_z

    return c, dc


# =============================================================================
# 2D NUFFT with custom gradients
# =============================================================================


def _nufft2d1_impl(
    x: Array, y: Array, c: Array, n_modes: tuple[int, int], eps: float = 1e-6, isign: int = 1
) -> Array:
    """Implementation of 2D Type 1 NUFFT."""
    from .nufft1 import nufft2d1 as nufft2d1_fast

    return nufft2d1_fast(x, y, c, n_modes, eps, isign)


def _nufft2d2_impl(x: Array, y: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """Implementation of 2D Type 2 NUFFT."""
    from .nufft2 import nufft2d2 as nufft2d2_fast

    return nufft2d2_fast(x, y, f, eps, isign)


@jax.custom_vjp
def nufft2d1(
    x: Array, y: Array, c: Array, n_modes: tuple[int, int], eps: float = 1e-6, isign: int = 1
) -> Array:
    """
    2D Type 1 NUFFT with custom gradients.

    Args:
        x: x-coordinates of nonuniform points, shape (M,)
        y: y-coordinates of nonuniform points, shape (M,)
        c: Complex strengths, shape (M,)
        n_modes: (n1, n2) number of modes in each dimension
        eps: Requested precision
        isign: Sign of exponent

    Returns:
        f: Fourier coefficients, shape (n2, n1)
    """
    return _nufft2d1_impl(x, y, c, n_modes, eps, isign)


def _nufft2d1_fwd(x, y, c, n_modes, eps, isign):
    f = _nufft2d1_impl(x, y, c, n_modes, eps, isign)
    return f, (x, y, c, n_modes, eps, isign)


def _nufft2d1_bwd(res, g):
    x, y, c, n_modes, eps, isign = res
    n1, n2 = n_modes

    # Conjugate g for correct gradient through holomorphic function
    # g has shape (n2, n1): y-axis is axis 0, x-axis is axis 1
    g_conj = jnp.conj(g)

    # Gradient w.r.t. c
    dc = _nufft2d2_impl(x, y, g_conj, eps, -isign)

    # Gradient w.r.t. x: k1 indexes axis 1 (size n1)
    k1 = jnp.arange(-n1 // 2, (n1 + 1) // 2)
    k1_g = g_conj * k1[None, :]  # Weight axis 1 by k1
    dx_complex = _nufft2d2_impl(x, y, k1_g, eps, -isign)
    dx = jnp.real(jnp.conj(c) * (-1j) * isign * dx_complex)

    # Gradient w.r.t. y: k2 indexes axis 0 (size n2)
    k2 = jnp.arange(-n2 // 2, (n2 + 1) // 2)
    k2_g = g_conj * k2[:, None]  # Weight axis 0 by k2
    dy_complex = _nufft2d2_impl(x, y, k2_g, eps, -isign)
    dy = jnp.real(jnp.conj(c) * (-1j) * isign * dy_complex)

    return (dx, dy, dc, None, None, None)


nufft2d1.defvjp(_nufft2d1_fwd, _nufft2d1_bwd)


@jax.custom_vjp
def nufft2d2(x: Array, y: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """
    2D Type 2 NUFFT with custom gradients.

    Args:
        x: x-coordinates of nonuniform points, shape (M,)
        y: y-coordinates of nonuniform points, shape (M,)
        f: Fourier coefficients, shape (n2, n1)
        eps: Requested precision
        isign: Sign of exponent

    Returns:
        c: Complex values at nonuniform points, shape (M,)
    """
    return _nufft2d2_impl(x, y, f, eps, isign)


def _nufft2d2_fwd(x, y, f, eps, isign):
    c = _nufft2d2_impl(x, y, f, eps, isign)
    return c, (x, y, f, eps, isign)


def _nufft2d2_bwd(res, g):
    x, y, f, eps, isign = res
    # f has shape (n2, n1): y-axis is axis 0, x-axis is axis 1
    n2, n1 = f.shape

    # Conjugate g for correct gradient through holomorphic function
    g_conj = jnp.conj(g)
    f_conj = jnp.conj(f)

    # Gradient w.r.t. f
    df = _nufft2d1_impl(x, y, g_conj, (n1, n2), eps, -isign)

    # Gradient w.r.t. x: k1 indexes axis 1 (size n1)
    # Use conj(f) and -isign for the correct gradient formula
    k1 = jnp.arange(-n1 // 2, (n1 + 1) // 2)
    k1_f = f_conj * k1[None, :]  # Weight axis 1 by k1
    dx_complex = _nufft2d2_impl(x, y, k1_f, eps, -isign)
    dx = jnp.real(g_conj * (-1j) * isign * dx_complex)

    # Gradient w.r.t. y: k2 indexes axis 0 (size n2)
    k2 = jnp.arange(-n2 // 2, (n2 + 1) // 2)
    k2_f = f_conj * k2[:, None]  # Weight axis 0 by k2
    dy_complex = _nufft2d2_impl(x, y, k2_f, eps, -isign)
    dy = jnp.real(g_conj * (-1j) * isign * dy_complex)

    return (dx, dy, df, None, None)


nufft2d2.defvjp(_nufft2d2_fwd, _nufft2d2_bwd)


# =============================================================================
# 3D NUFFT with custom gradients
# =============================================================================


def _nufft3d1_impl(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
) -> Array:
    """Implementation of 3D Type 1 NUFFT."""
    from .nufft1 import nufft3d1 as nufft3d1_fast

    return nufft3d1_fast(x, y, z, c, n_modes, eps, isign)


def _nufft3d2_impl(
    x: Array, y: Array, z: Array, f: Array, eps: float = 1e-6, isign: int = -1
) -> Array:
    """Implementation of 3D Type 2 NUFFT."""
    from .nufft2 import nufft3d2 as nufft3d2_fast

    return nufft3d2_fast(x, y, z, f, eps, isign)


@jax.custom_vjp
def nufft3d1(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
) -> Array:
    """
    3D Type 1 NUFFT with custom gradients.

    Args:
        x, y, z: Coordinates of nonuniform points, each shape (M,)
        c: Complex strengths, shape (M,)
        n_modes: (n1, n2, n3) number of modes
        eps: Requested precision
        isign: Sign of exponent

    Returns:
        f: Fourier coefficients, shape (n1, n2, n3)
    """
    return _nufft3d1_impl(x, y, z, c, n_modes, eps, isign)


def _nufft3d1_fwd(x, y, z, c, n_modes, eps, isign):
    f = _nufft3d1_impl(x, y, z, c, n_modes, eps, isign)
    return f, (x, y, z, c, n_modes, eps, isign)


def _nufft3d1_bwd(res, g):
    x, y, z, c, n_modes, eps, isign = res
    n1, n2, n3 = n_modes

    # Conjugate g for correct gradient through holomorphic function
    # g has shape (n3, n2, n1): z-axis is axis 0, y-axis is axis 1, x-axis is axis 2
    g_conj = jnp.conj(g)

    # Gradient w.r.t. c
    dc = _nufft3d2_impl(x, y, z, g_conj, eps, -isign)

    # Gradient w.r.t. x: k1 indexes axis 2 (size n1)
    k1 = jnp.arange(-n1 // 2, (n1 + 1) // 2)
    k1_g = g_conj * k1[None, None, :]  # Weight axis 2 by k1
    dx_complex = _nufft3d2_impl(x, y, z, k1_g, eps, -isign)
    dx = jnp.real(jnp.conj(c) * (-1j) * isign * dx_complex)

    # Gradient w.r.t. y: k2 indexes axis 1 (size n2)
    k2 = jnp.arange(-n2 // 2, (n2 + 1) // 2)
    k2_g = g_conj * k2[None, :, None]  # Weight axis 1 by k2
    dy_complex = _nufft3d2_impl(x, y, z, k2_g, eps, -isign)
    dy = jnp.real(jnp.conj(c) * (-1j) * isign * dy_complex)

    # Gradient w.r.t. z: k3 indexes axis 0 (size n3)
    k3 = jnp.arange(-n3 // 2, (n3 + 1) // 2)
    k3_g = g_conj * k3[:, None, None]  # Weight axis 0 by k3
    dz_complex = _nufft3d2_impl(x, y, z, k3_g, eps, -isign)
    dz = jnp.real(jnp.conj(c) * (-1j) * isign * dz_complex)

    return (dx, dy, dz, dc, None, None, None)


nufft3d1.defvjp(_nufft3d1_fwd, _nufft3d1_bwd)


@jax.custom_vjp
def nufft3d2(x: Array, y: Array, z: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """
    3D Type 2 NUFFT with custom gradients.

    Args:
        x, y, z: Coordinates of nonuniform points, each shape (M,)
        f: Fourier coefficients, shape (n1, n2, n3)
        eps: Requested precision
        isign: Sign of exponent

    Returns:
        c: Complex values at nonuniform points, shape (M,)
    """
    return _nufft3d2_impl(x, y, z, f, eps, isign)


def _nufft3d2_fwd(x, y, z, f, eps, isign):
    c = _nufft3d2_impl(x, y, z, f, eps, isign)
    return c, (x, y, z, f, eps, isign)


def _nufft3d2_bwd(res, g):
    x, y, z, f, eps, isign = res
    # f has shape (n3, n2, n1): z-axis is axis 0, y-axis is axis 1, x-axis is axis 2
    n3, n2, n1 = f.shape

    # Conjugate g for correct gradient through holomorphic function
    g_conj = jnp.conj(g)
    f_conj = jnp.conj(f)

    # Gradient w.r.t. f
    df = _nufft3d1_impl(x, y, z, g_conj, (n1, n2, n3), eps, -isign)

    # Gradient w.r.t. x: k1 indexes axis 2 (size n1)
    # Use conj(f) and -isign for correct gradient formula
    k1 = jnp.arange(-n1 // 2, (n1 + 1) // 2)
    k1_f = f_conj * k1[None, None, :]  # Weight axis 2 by k1
    dx_complex = _nufft3d2_impl(x, y, z, k1_f, eps, -isign)
    dx = jnp.real(g_conj * (-1j) * isign * dx_complex)

    # Gradient w.r.t. y: k2 indexes axis 1 (size n2)
    k2 = jnp.arange(-n2 // 2, (n2 + 1) // 2)
    k2_f = f_conj * k2[None, :, None]  # Weight axis 1 by k2
    dy_complex = _nufft3d2_impl(x, y, z, k2_f, eps, -isign)
    dy = jnp.real(g_conj * (-1j) * isign * dy_complex)

    # Gradient w.r.t. z: k3 indexes axis 0 (size n3)
    k3 = jnp.arange(-n3 // 2, (n3 + 1) // 2)
    k3_f = f_conj * k3[:, None, None]  # Weight axis 0 by k3
    dz_complex = _nufft3d2_impl(x, y, z, k3_f, eps, -isign)
    dz = jnp.real(g_conj * (-1j) * isign * dz_complex)

    return (dx, dy, dz, df, None, None)


nufft3d2.defvjp(_nufft3d2_fwd, _nufft3d2_bwd)


# =============================================================================
# Helper functions for computing gradients w.r.t. point positions
# =============================================================================


def compute_position_gradient_1d(
    x: Array,
    c: Array,
    upstream_grad: Array,
    n_modes: int,
    eps: float,
    isign: int,
    transform_type: int,
) -> Array:
    """
    Compute gradient of NUFFT w.r.t. point positions x.

    This helper abstracts the common pattern for computing position gradients
    in both Type 1 and Type 2 transforms.

    Args:
        x: Point positions, shape (M,)
        c: Strengths or interpolated values (depends on transform type)
        upstream_grad: Upstream gradient (g)
        n_modes: Number of Fourier modes
        eps: Precision tolerance
        isign: Sign of exponent in the transform
        transform_type: 1 for Type 1, 2 for Type 2

    Returns:
        dx: Gradient w.r.t. x, shape (M,)
    """
    k = jnp.arange(-(n_modes // 2), (n_modes + 1) // 2)
    g_conj = jnp.conj(upstream_grad)

    if transform_type == 1:
        # For Type 1: f[k] = sum_j c[j] * exp(i * isign * k * x[j])
        kg = k * g_conj
        dx_complex = _nufft1d2_impl(x, kg, eps, -isign)
        dx = jnp.real(jnp.conj(c) * (-1j) * isign * dx_complex)
    else:
        # For Type 2: c[j] = sum_k f[k] * exp(i * isign * k * x[j])
        # Note: c here is actually f (the Fourier coefficients)
        # Use conj(c) and -isign for correct gradient formula
        kf = k * jnp.conj(c)  # c is f for Type 2
        dx_complex = _nufft1d2_impl(x, kf, eps, -isign)
        dx = jnp.real(g_conj * (-1j) * isign * dx_complex)

    return dx


def compute_position_gradient_2d(
    x: Array,
    y: Array,
    weights: Array,
    upstream_grad: Array,
    n_modes: tuple[int, int],
    eps: float,
    isign: int,
    transform_type: int,
) -> tuple[Array, Array]:
    """
    Compute gradients of 2D NUFFT w.r.t. point positions (x, y).

    Args:
        x, y: Point positions, each shape (M,)
        weights: c for Type 1, f for Type 2
        upstream_grad: Upstream gradient, shape (n2, n1)
        n_modes: (n1, n2) number of modes
        eps: Precision tolerance
        isign: Sign of exponent
        transform_type: 1 for Type 1, 2 for Type 2

    Returns:
        (dx, dy): Gradients w.r.t. x and y, each shape (M,)
    """
    n1, n2 = n_modes
    k1 = jnp.arange(-n1 // 2, (n1 + 1) // 2)
    k2 = jnp.arange(-n2 // 2, (n2 + 1) // 2)
    g_conj = jnp.conj(upstream_grad)

    if transform_type == 1:
        # g is upstream gradient with shape (n2, n1): y-axis is axis 0, x-axis is axis 1
        k1_g = g_conj * k1[None, :]  # Weight axis 1 by k1
        k2_g = g_conj * k2[:, None]  # Weight axis 0 by k2

        dx_complex = _nufft2d2_impl(x, y, k1_g, eps, -isign)
        dy_complex = _nufft2d2_impl(x, y, k2_g, eps, -isign)

        dx = jnp.real(jnp.conj(weights) * (-1j) * isign * dx_complex)
        dy = jnp.real(jnp.conj(weights) * (-1j) * isign * dy_complex)
    else:
        # weights is f (Fourier coefficients) with shape (n2, n1)
        # Use conj(weights) and -isign for correct gradient formula
        w_conj = jnp.conj(weights)
        k1_f = w_conj * k1[None, :]  # Weight axis 1 by k1
        k2_f = w_conj * k2[:, None]  # Weight axis 0 by k2

        dx_complex = _nufft2d2_impl(x, y, k1_f, eps, -isign)
        dy_complex = _nufft2d2_impl(x, y, k2_f, eps, -isign)

        dx = jnp.real(g_conj * (-1j) * isign * dx_complex)
        dy = jnp.real(g_conj * (-1j) * isign * dy_complex)

    return dx, dy


def compute_position_gradient_3d(
    x: Array,
    y: Array,
    z: Array,
    weights: Array,
    upstream_grad: Array,
    n_modes: tuple[int, int, int],
    eps: float,
    isign: int,
    transform_type: int,
) -> tuple[Array, Array, Array]:
    """
    Compute gradients of 3D NUFFT w.r.t. point positions (x, y, z).

    Args:
        x, y, z: Point positions, each shape (M,)
        weights: c for Type 1, f for Type 2
        upstream_grad: Upstream gradient, shape (n3, n2, n1)
        n_modes: (n1, n2, n3) number of modes
        eps: Precision tolerance
        isign: Sign of exponent
        transform_type: 1 for Type 1, 2 for Type 2

    Returns:
        (dx, dy, dz): Gradients w.r.t. x, y, z, each shape (M,)
    """
    n1, n2, n3 = n_modes
    k1 = jnp.arange(-n1 // 2, (n1 + 1) // 2)
    k2 = jnp.arange(-n2 // 2, (n2 + 1) // 2)
    k3 = jnp.arange(-n3 // 2, (n3 + 1) // 2)
    g_conj = jnp.conj(upstream_grad)

    if transform_type == 1:
        # g has shape (n3, n2, n1): z-axis is axis 0, y-axis is axis 1, x-axis is axis 2
        k1_g = g_conj * k1[None, None, :]  # Weight axis 2 by k1
        k2_g = g_conj * k2[None, :, None]  # Weight axis 1 by k2
        k3_g = g_conj * k3[:, None, None]  # Weight axis 0 by k3

        dx_complex = _nufft3d2_impl(x, y, z, k1_g, eps, -isign)
        dy_complex = _nufft3d2_impl(x, y, z, k2_g, eps, -isign)
        dz_complex = _nufft3d2_impl(x, y, z, k3_g, eps, -isign)

        dx = jnp.real(jnp.conj(weights) * (-1j) * isign * dx_complex)
        dy = jnp.real(jnp.conj(weights) * (-1j) * isign * dy_complex)
        dz = jnp.real(jnp.conj(weights) * (-1j) * isign * dz_complex)
    else:
        # weights (f) has shape (n3, n2, n1)
        # Use conj(weights) and -isign for correct gradient formula
        w_conj = jnp.conj(weights)
        k1_f = w_conj * k1[None, None, :]  # Weight axis 2 by k1
        k2_f = w_conj * k2[None, :, None]  # Weight axis 1 by k2
        k3_f = w_conj * k3[:, None, None]  # Weight axis 0 by k3

        dx_complex = _nufft3d2_impl(x, y, z, k1_f, eps, -isign)
        dy_complex = _nufft3d2_impl(x, y, z, k2_f, eps, -isign)
        dz_complex = _nufft3d2_impl(x, y, z, k3_f, eps, -isign)

        dx = jnp.real(g_conj * (-1j) * isign * dx_complex)
        dy = jnp.real(g_conj * (-1j) * isign * dy_complex)
        dz = jnp.real(g_conj * (-1j) * isign * dz_complex)

    return dx, dy, dz


# =============================================================================
# Spread/Interpolation gradients (lower-level)
# =============================================================================


def spread_grad_x_1d(
    x: Array,
    c: Array,
    g: Array,
    kernel_params,
) -> Array:
    """
    Gradient of spread operation w.r.t. point positions.

    Uses kernel derivative: d/dx phi((k-x)/gamma) = -1/gamma * phi'((k-x)/gamma)

    Args:
        x: Point positions (scaled to grid coordinates)
        c: Strengths at points
        g: Upstream gradient on the grid
        kernel_params: KernelParams instance

    Returns:
        dx: Gradient w.r.t. x
    """
    from ..core.kernel import es_kernel_derivative

    nspread = kernel_params.nspread
    beta = kernel_params.beta
    c_param = kernel_params.c
    nf = g.shape[0]

    # Half-width
    ns2 = nspread // 2

    def compute_dx_single(xi, ci):
        """Compute gradient for a single point."""
        # Find center grid point
        i_center = jnp.floor(xi).astype(jnp.int32)

        # Distance from center
        xi_frac = xi - i_center

        # Grid indices to consider
        i_vals = jnp.arange(-ns2 + 1, ns2 + 1) + i_center

        # Wrap around for periodic boundary
        i_vals_wrapped = i_vals % nf

        # Kernel argument
        z = (jnp.arange(-ns2 + 1, ns2 + 1) - xi_frac) * 2.0 / nspread

        # Kernel derivative at these points
        dphi = es_kernel_derivative(z, beta, c_param)

        # Chain rule: d/dx_i spread = -sum_k g[k] * dphi(k-x) * c
        # The -2/nspread factor comes from the chain rule on the normalization
        dx = -2.0 / nspread * jnp.sum(g[i_vals_wrapped] * dphi) * jnp.conj(ci)

        return jnp.real(dx)

    # Vectorize over all points
    dx = jax.vmap(compute_dx_single)(x, c)

    return dx


# =============================================================================
# Wrapper factory for adding gradients to existing functions
# =============================================================================


def with_custom_vjp(
    impl_fn: Callable,
    fwd_fn: Callable,
    bwd_fn: Callable,
) -> Callable:
    """
    Factory function to create a function with custom VJP.

    Args:
        impl_fn: The implementation function
        fwd_fn: Forward pass that returns (output, residuals)
        bwd_fn: Backward pass that computes gradients

    Returns:
        A new function with custom VJP defined
    """

    @jax.custom_vjp
    def wrapped(*args, **kwargs):
        return impl_fn(*args, **kwargs)

    wrapped.defvjp(fwd_fn, bwd_fn)
    return wrapped


def with_custom_jvp(
    impl_fn: Callable,
    jvp_fn: Callable,
) -> Callable:
    """
    Factory function to create a function with custom JVP.

    Args:
        impl_fn: The implementation function
        jvp_fn: JVP function that computes (primal_out, tangent_out)

    Returns:
        A new function with custom JVP defined
    """

    @jax.custom_jvp
    def wrapped(*args, **kwargs):
        return impl_fn(*args, **kwargs)

    @wrapped.defjvp
    def wrapped_jvp(primals, tangents):
        return jvp_fn(primals, tangents)

    return wrapped


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # 1D transforms with VJP
    "nufft1d1",
    "nufft1d2",
    # 1D transforms with JVP
    "nufft1d1_jvp",
    "nufft1d2_jvp",
    # 2D transforms with VJP
    "nufft2d1",
    "nufft2d2",
    # 2D transforms with JVP
    "nufft2d1_jvp",
    "nufft2d2_jvp",
    # 3D transforms with VJP
    "nufft3d1",
    "nufft3d2",
    # 3D transforms with JVP
    "nufft3d1_jvp",
    "nufft3d2_jvp",
    # Helper functions
    "compute_position_gradient_1d",
    "compute_position_gradient_2d",
    "compute_position_gradient_3d",
    # Factories
    "with_custom_vjp",
    "with_custom_jvp",
]
