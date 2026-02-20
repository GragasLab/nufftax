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
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Internal implementation wrappers
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


def _nufft1d1_bwd(n_modes: int, eps: float, isign: int, res: tuple, g: Array) -> tuple[Array | None, Array | None]:
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


def _nufft1d2_bwd(eps: float, isign: int, res: tuple, g: Array) -> tuple[Array | None, Array | None]:
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
def nufft2d1_jvp(x: Array, y: Array, c: Array, n_modes: tuple[int, int], eps: float = 1e-6, isign: int = 1) -> Array:
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
    df_from_c = _nufft2d1_impl(x, y, dc, n_modes, eps, isign) if dc is not None else jnp.zeros_like(f)

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
    df_from_c = _nufft3d1_impl(x, y, z, dc, n_modes, eps, isign) if dc is not None else jnp.zeros_like(f)

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
def nufft3d2_jvp(x: Array, y: Array, z: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
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


def _nufft2d1_impl(x: Array, y: Array, c: Array, n_modes: tuple[int, int], eps: float = 1e-6, isign: int = 1) -> Array:
    """Implementation of 2D Type 1 NUFFT."""
    from .nufft1 import nufft2d1 as nufft2d1_fast

    return nufft2d1_fast(x, y, c, n_modes, eps, isign)


def _nufft2d2_impl(x: Array, y: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """Implementation of 2D Type 2 NUFFT."""
    from .nufft2 import nufft2d2 as nufft2d2_fast

    return nufft2d2_fast(x, y, f, eps, isign)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def nufft2d1(x: Array, y: Array, c: Array, n_modes: tuple[int, int], eps: float = 1e-6, isign: int = 1) -> Array:
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

    Note: With nondiff_argnums=(3, 4, 5), the signature is:
    (diff_args..., nondiff_args...) = (x, y, c, n_modes, eps, isign)
    """
    return _nufft2d1_impl(x, y, c, n_modes, eps, isign)


def _nufft2d1_fwd(x, y, c, n_modes, eps, isign):
    f = _nufft2d1_impl(x, y, c, n_modes, eps, isign)
    # Only save diff args (x, y, c)
    return f, (x, y, c)


def _nufft2d1_bwd(n_modes, eps, isign, res, g):
    x, y, c = res
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

    return (dx, dy, dc)


nufft2d1.defvjp(_nufft2d1_fwd, _nufft2d1_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
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

    Note: With nondiff_argnums=(3, 4), the signature is:
    (diff_args..., nondiff_args...) = (x, y, f, eps, isign)
    """
    return _nufft2d2_impl(x, y, f, eps, isign)


def _nufft2d2_fwd(x, y, f, eps, isign):
    c = _nufft2d2_impl(x, y, f, eps, isign)
    # Only save diff args (x, y, f)
    return c, (x, y, f)


def _nufft2d2_bwd(eps, isign, res, g):
    x, y, f = res
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

    return (dx, dy, df)


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


def _nufft3d2_impl(x: Array, y: Array, z: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    """Implementation of 3D Type 2 NUFFT."""
    from .nufft2 import nufft3d2 as nufft3d2_fast

    return nufft3d2_fast(x, y, z, f, eps, isign)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
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

    Note: With nondiff_argnums=(4, 5, 6), the signature is:
    (diff_args..., nondiff_args...) = (x, y, z, c, n_modes, eps, isign)
    """
    return _nufft3d1_impl(x, y, z, c, n_modes, eps, isign)


def _nufft3d1_fwd(x, y, z, c, n_modes, eps, isign):
    f = _nufft3d1_impl(x, y, z, c, n_modes, eps, isign)
    # Only save diff args (x, y, z, c)
    return f, (x, y, z, c)


def _nufft3d1_bwd(n_modes, eps, isign, res, g):
    x, y, z, c = res
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

    return (dx, dy, dz, dc)


nufft3d1.defvjp(_nufft3d1_fwd, _nufft3d1_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
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

    Note: With nondiff_argnums=(4, 5), the signature is:
    (diff_args..., nondiff_args...) = (x, y, z, f, eps, isign)
    """
    return _nufft3d2_impl(x, y, z, f, eps, isign)


def _nufft3d2_fwd(x, y, z, f, eps, isign):
    c = _nufft3d2_impl(x, y, z, f, eps, isign)
    # Only save diff args (x, y, z, f)
    return c, (x, y, z, f)


def _nufft3d2_bwd(eps, isign, res, g):
    x, y, z, f = res
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

    return (dx, dy, dz, df)


nufft3d2.defvjp(_nufft3d2_fwd, _nufft3d2_bwd)


# =============================================================================
# Type 3 NUFFT implementations
# =============================================================================


def _nufft1d3_impl(
    x: Array,
    c: Array,
    s: Array,
    n_modes: int,
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """Implementation of 1D Type 3 NUFFT."""
    from .nufft3 import nufft1d3 as nufft1d3_fast

    return nufft1d3_fast(x, c, s, n_modes, eps, isign, upsampfac)


def _nufft2d3_impl(
    x: Array,
    y: Array,
    c: Array,
    s: Array,
    t: Array,
    n_modes: tuple[int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """Implementation of 2D Type 3 NUFFT."""
    from .nufft3 import nufft2d3 as nufft2d3_fast

    return nufft2d3_fast(x, y, c, s, t, n_modes, eps, isign, upsampfac)


def _nufft3d3_impl(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    s: Array,
    t: Array,
    u: Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """Implementation of 3D Type 3 NUFFT."""
    from .nufft3 import nufft3d3 as nufft3d3_fast

    return nufft3d3_fast(x, y, z, c, s, t, u, n_modes, eps, isign, upsampfac)


# =============================================================================
# 1D Type 3 NUFFT with custom VJP
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def nufft1d3(
    x: Array,
    c: Array,
    s: Array,
    n_modes: int,
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """
    1D Type 3 NUFFT with custom gradients.

    Computes:
        f[k] = sum_j c[j] * exp(isign * i * s[k] * x[j])

    Args:
        x: Source point locations, shape (M,)
        c: Complex strengths at source points, shape (M,) or (n_trans, M)
        s: Target frequencies (nonuniform), shape (N,)
        n_modes: Grid size for internal computation
        eps: Requested precision (tolerance)
        isign: Sign of exponent (+1 or -1)
        upsampfac: Oversampling factor

    Returns:
        f: Values at target frequencies, shape (N,) or (n_trans, N)
    """
    return _nufft1d3_impl(x, c, s, n_modes, eps, isign, upsampfac)


def _nufft1d3_fwd(
    x: Array,
    c: Array,
    s: Array,
    n_modes: int,
    eps: float,
    isign: int,
    upsampfac: float,
) -> tuple[Array, tuple]:
    """Forward pass for nufft1d3 VJP."""
    f = _nufft1d3_impl(x, c, s, n_modes, eps, isign, upsampfac)
    return f, (x, c, s)


def _nufft1d3_bwd(
    n_modes: int,
    eps: float,
    isign: int,
    upsampfac: float,
    res: tuple,
    g: Array,
) -> tuple[Array, Array, Array]:
    """
    Backward pass for nufft1d3 VJP.

    For f[k] = sum_j c[j] * exp(isign * i * s[k] * x[j]):
    - grad w.r.t. c: Type 3 from s to x with conj(g)
    - grad w.r.t. x: s-weighted Type 3
    - grad w.r.t. s: x-weighted Type 3 (element-wise)

    Note: JAX passes g as the cotangent dL/d(conj(f)). For L = |f|^2, this means g = 2*conj(f).
    The correct formula for real parameters is Re(g * df/ds), not Re(conj(g) * df/ds).
    """
    from .nufft3 import compute_type3_grid_size

    x, c, s = res

    # Handle batched input
    batched = c.ndim == 2
    if batched:
        g_conj = jnp.conj(g)  # (n_trans, N)
        g_batched = g
    else:
        g_conj = jnp.conj(g)  # (N,)
        g_batched = g[None, :]  # (1, N)
        c = c[None, :]
        g_conj = g_conj[None, :]

    # Compute grid size for reverse transform (s -> x)
    n_modes_rev = compute_type3_grid_size(s, x, eps, upsampfac)

    # Gradient w.r.t. c: "reverse" Type 3 from s to x
    # dc[j] = sum_k conj(g[k]) * exp(-isign * i * s[k] * x[j])
    # This is nufft1d3 with: source=s, target=x, coeff=conj(g), sign=-isign
    dc = _nufft1d3_impl(s, g_conj, x, n_modes_rev, eps, -isign, upsampfac)

    # Gradient w.r.t. x:
    # dx[j] = Re(conj(c[j]) * (-1j * isign) * sum_k s[k] * conj(g[k]) * exp(-isign * i * s[k] * x[j]))
    s_weighted_g = s[None, :] * g_conj  # (n_trans, N) or (1, N)
    dx_complex = _nufft1d3_impl(s, s_weighted_g, x, n_modes_rev, eps, -isign, upsampfac)
    dx = jnp.real(jnp.conj(c) * (-1j) * isign * dx_complex)

    # Gradient w.r.t. s:
    # df[k]/ds[k] = 1j * isign * sum_j x[j] * c[j] * exp(isign * i * s[k] * x[j])
    # For real parameter s, dL/ds = Re(g * df/ds)
    x_weighted_c = x[None, :] * c  # (n_trans, M) or (1, M)
    ds_complex = _nufft1d3_impl(x, x_weighted_c, s, n_modes, eps, isign, upsampfac)
    ds = jnp.real(g_batched * 1j * isign * ds_complex)

    # Sum over batch dimension if batched
    if batched:
        dx = jnp.sum(dx, axis=0)
        ds = jnp.sum(ds, axis=0)
    else:
        dc = dc[0]
        dx = dx[0]
        ds = ds[0]

    return (dx, dc, ds)


nufft1d3.defvjp(_nufft1d3_fwd, _nufft1d3_bwd)


# =============================================================================
# 2D Type 3 NUFFT with custom VJP
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def nufft2d3(
    x: Array,
    y: Array,
    c: Array,
    s: Array,
    t: Array,
    n_modes: tuple[int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """
    2D Type 3 NUFFT with custom gradients.

    Computes:
        f[k] = sum_j c[j] * exp(isign * i * (s[k]*x[j] + t[k]*y[j]))

    Args:
        x, y: Source point locations, each shape (M,)
        c: Complex strengths, shape (M,) or (n_trans, M)
        s, t: Target frequencies (nonuniform), each shape (N,)
        n_modes: Grid sizes (nf1, nf2) for internal computation
        eps: Requested precision
        isign: Sign of exponent
        upsampfac: Oversampling factor

    Returns:
        f: Values at target frequencies, shape (N,) or (n_trans, N)
    """
    return _nufft2d3_impl(x, y, c, s, t, n_modes, eps, isign, upsampfac)


def _nufft2d3_fwd(
    x: Array,
    y: Array,
    c: Array,
    s: Array,
    t: Array,
    n_modes: tuple[int, int],
    eps: float,
    isign: int,
    upsampfac: float,
) -> tuple[Array, tuple]:
    """Forward pass for nufft2d3 VJP."""
    f = _nufft2d3_impl(x, y, c, s, t, n_modes, eps, isign, upsampfac)
    return f, (x, y, c, s, t)


def _nufft2d3_bwd(
    n_modes: tuple[int, int],
    eps: float,
    isign: int,
    upsampfac: float,
    res: tuple,
    g: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Backward pass for nufft2d3 VJP."""
    from .nufft3 import compute_type3_grid_sizes_2d

    x, y, c, s, t = res

    # Handle batched input
    batched = c.ndim == 2
    if batched:
        g_conj = jnp.conj(g)
        g_batched = g
    else:
        g_conj = jnp.conj(g)
        g_batched = g[None, :]
        c = c[None, :]
        g_conj = g_conj[None, :]

    # Compute extents for reverse transform
    s_ext = float((jnp.max(s) - jnp.min(s)) / 2)
    t_ext = float((jnp.max(t) - jnp.min(t)) / 2)
    x_ext = float((jnp.max(x) - jnp.min(x)) / 2)
    y_ext = float((jnp.max(y) - jnp.min(y)) / 2)
    n_modes_rev = compute_type3_grid_sizes_2d(s_ext, t_ext, x_ext, y_ext, eps, upsampfac)

    # Gradient w.r.t. c: reverse Type 3 from (s,t) to (x,y)
    dc = _nufft2d3_impl(s, t, g_conj, x, y, n_modes_rev, eps, -isign, upsampfac)

    # Gradient w.r.t. x: s-weighted
    s_weighted_g = s[None, :] * g_conj
    dx_complex = _nufft2d3_impl(s, t, s_weighted_g, x, y, n_modes_rev, eps, -isign, upsampfac)
    dx = jnp.real(jnp.conj(c) * (-1j) * isign * dx_complex)

    # Gradient w.r.t. y: t-weighted
    t_weighted_g = t[None, :] * g_conj
    dy_complex = _nufft2d3_impl(s, t, t_weighted_g, x, y, n_modes_rev, eps, -isign, upsampfac)
    dy = jnp.real(jnp.conj(c) * (-1j) * isign * dy_complex)

    # Gradient w.r.t. s: x-weighted (use g, not g_conj, for real parameters)
    x_weighted_c = x[None, :] * c
    ds_complex = _nufft2d3_impl(x, y, x_weighted_c, s, t, n_modes, eps, isign, upsampfac)
    ds = jnp.real(g_batched * 1j * isign * ds_complex)

    # Gradient w.r.t. t: y-weighted (use g, not g_conj, for real parameters)
    y_weighted_c = y[None, :] * c
    dt_complex = _nufft2d3_impl(x, y, y_weighted_c, s, t, n_modes, eps, isign, upsampfac)
    dt = jnp.real(g_batched * 1j * isign * dt_complex)

    # Sum over batch dimension if batched
    if batched:
        dx = jnp.sum(dx, axis=0)
        dy = jnp.sum(dy, axis=0)
        ds = jnp.sum(ds, axis=0)
        dt = jnp.sum(dt, axis=0)
    else:
        dc = dc[0]
        dx = dx[0]
        dy = dy[0]
        ds = ds[0]
        dt = dt[0]

    return (dx, dy, dc, ds, dt)


nufft2d3.defvjp(_nufft2d3_fwd, _nufft2d3_bwd)


# =============================================================================
# 3D Type 3 NUFFT with custom VJP
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10))
def nufft3d3(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    s: Array,
    t: Array,
    u: Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """
    3D Type 3 NUFFT with custom gradients.

    Computes:
        f[k] = sum_j c[j] * exp(isign * i * (s[k]*x[j] + t[k]*y[j] + u[k]*z[j]))

    Args:
        x, y, z: Source point locations, each shape (M,)
        c: Complex strengths, shape (M,) or (n_trans, M)
        s, t, u: Target frequencies (nonuniform), each shape (N,)
        n_modes: Grid sizes (nf1, nf2, nf3) for internal computation
        eps: Requested precision
        isign: Sign of exponent
        upsampfac: Oversampling factor

    Returns:
        f: Values at target frequencies, shape (N,) or (n_trans, N)
    """
    return _nufft3d3_impl(x, y, z, c, s, t, u, n_modes, eps, isign, upsampfac)


def _nufft3d3_fwd(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    s: Array,
    t: Array,
    u: Array,
    n_modes: tuple[int, int, int],
    eps: float,
    isign: int,
    upsampfac: float,
) -> tuple[Array, tuple]:
    """Forward pass for nufft3d3 VJP."""
    f = _nufft3d3_impl(x, y, z, c, s, t, u, n_modes, eps, isign, upsampfac)
    return f, (x, y, z, c, s, t, u)


def _nufft3d3_bwd(
    n_modes: tuple[int, int, int],
    eps: float,
    isign: int,
    upsampfac: float,
    res: tuple,
    g: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Backward pass for nufft3d3 VJP."""
    from .nufft3 import compute_type3_grid_sizes_3d

    x, y, z, c, s, t, u = res

    # Handle batched input
    batched = c.ndim == 2
    if batched:
        g_conj = jnp.conj(g)
        g_batched = g
    else:
        g_conj = jnp.conj(g)
        g_batched = g[None, :]
        c = c[None, :]
        g_conj = g_conj[None, :]

    # Compute extents for reverse transform
    s_ext = float((jnp.max(s) - jnp.min(s)) / 2)
    t_ext = float((jnp.max(t) - jnp.min(t)) / 2)
    u_ext = float((jnp.max(u) - jnp.min(u)) / 2)
    x_ext = float((jnp.max(x) - jnp.min(x)) / 2)
    y_ext = float((jnp.max(y) - jnp.min(y)) / 2)
    z_ext = float((jnp.max(z) - jnp.min(z)) / 2)
    n_modes_rev = compute_type3_grid_sizes_3d(s_ext, t_ext, u_ext, x_ext, y_ext, z_ext, eps, upsampfac)

    # Gradient w.r.t. c: reverse Type 3
    dc = _nufft3d3_impl(s, t, u, g_conj, x, y, z, n_modes_rev, eps, -isign, upsampfac)

    # Gradient w.r.t. x: s-weighted
    s_weighted_g = s[None, :] * g_conj
    dx_complex = _nufft3d3_impl(s, t, u, s_weighted_g, x, y, z, n_modes_rev, eps, -isign, upsampfac)
    dx = jnp.real(jnp.conj(c) * (-1j) * isign * dx_complex)

    # Gradient w.r.t. y: t-weighted
    t_weighted_g = t[None, :] * g_conj
    dy_complex = _nufft3d3_impl(s, t, u, t_weighted_g, x, y, z, n_modes_rev, eps, -isign, upsampfac)
    dy = jnp.real(jnp.conj(c) * (-1j) * isign * dy_complex)

    # Gradient w.r.t. z: u-weighted
    u_weighted_g = u[None, :] * g_conj
    dz_complex = _nufft3d3_impl(s, t, u, u_weighted_g, x, y, z, n_modes_rev, eps, -isign, upsampfac)
    dz = jnp.real(jnp.conj(c) * (-1j) * isign * dz_complex)

    # Gradient w.r.t. s: x-weighted (use g, not g_conj, for real parameters)
    x_weighted_c = x[None, :] * c
    ds_complex = _nufft3d3_impl(x, y, z, x_weighted_c, s, t, u, n_modes, eps, isign, upsampfac)
    ds = jnp.real(g_batched * 1j * isign * ds_complex)

    # Gradient w.r.t. t: y-weighted (use g, not g_conj, for real parameters)
    y_weighted_c = y[None, :] * c
    dt_complex = _nufft3d3_impl(x, y, z, y_weighted_c, s, t, u, n_modes, eps, isign, upsampfac)
    dt = jnp.real(g_batched * 1j * isign * dt_complex)

    # Gradient w.r.t. u: z-weighted (use g, not g_conj, for real parameters)
    z_weighted_c = z[None, :] * c
    du_complex = _nufft3d3_impl(x, y, z, z_weighted_c, s, t, u, n_modes, eps, isign, upsampfac)
    du = jnp.real(g_batched * 1j * isign * du_complex)

    # Sum over batch dimension if batched
    if batched:
        dx = jnp.sum(dx, axis=0)
        dy = jnp.sum(dy, axis=0)
        dz = jnp.sum(dz, axis=0)
        ds = jnp.sum(ds, axis=0)
        dt = jnp.sum(dt, axis=0)
        du = jnp.sum(du, axis=0)
    else:
        dc = dc[0]
        dx = dx[0]
        dy = dy[0]
        dz = dz[0]
        ds = ds[0]
        dt = dt[0]
        du = du[0]

    return (dx, dy, dz, dc, ds, dt, du)


nufft3d3.defvjp(_nufft3d3_fwd, _nufft3d3_bwd)


# =============================================================================
# 1D Type 3 NUFFT with custom JVP
# =============================================================================


@jax.custom_jvp
def nufft1d3_jvp(
    x: Array,
    c: Array,
    s: Array,
    n_modes: int,
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """1D Type 3 NUFFT with forward-mode AD support."""
    return _nufft1d3_impl(x, c, s, n_modes, eps, isign, upsampfac)


@nufft1d3_jvp.defjvp
def _nufft1d3_jvp_rule(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """Forward-mode differentiation for nufft1d3."""
    x, c, s, n_modes, eps, isign, upsampfac = primals
    dx, dc, ds, _, _, _, _ = tangents

    # Primal output
    f = _nufft1d3_impl(x, c, s, n_modes, eps, isign, upsampfac)

    # Tangent output: df = ∂f/∂c * dc + ∂f/∂x * dx + ∂f/∂s * ds

    # Contribution from dc: Type 3 of dc
    df_from_c = _nufft1d3_impl(x, dc, s, n_modes, eps, isign, upsampfac) if dc is not None else jnp.zeros_like(f)

    # Contribution from dx:
    # f[k] = sum_j c[j] * exp(isign * i * s[k] * x[j])
    # ∂f[k]/∂x[j] = isign * i * s[k] * c[j] * exp(isign * i * s[k] * x[j])
    # df[k] = isign * i * s[k] * sum_j (c[j] * dx[j]) * exp(isign * i * s[k] * x[j])
    #       = isign * i * s[k] * nufft1d3(x, c*dx, s)
    if dx is not None:
        weighted_transform = _nufft1d3_impl(x, c * dx, s, n_modes, eps, isign, upsampfac)
        df_from_x = 1j * isign * s * weighted_transform
    else:
        df_from_x = jnp.zeros_like(f)

    # Contribution from ds:
    # ∂f[k]/∂s[k] = isign * i * (sum_j x[j] * c[j] * exp(isign * i * s[k] * x[j]))
    # df[k] = isign * i * ds[k] * sum_j x[j] * c[j] * exp(...)
    if ds is not None:
        x_weighted = _nufft1d3_impl(x, x * c, s, n_modes, eps, isign, upsampfac)
        df_from_s = 1j * isign * ds * x_weighted
    else:
        df_from_s = jnp.zeros_like(f)

    df = df_from_c + df_from_x + df_from_s

    return f, df


# =============================================================================
# 2D Type 3 NUFFT with custom JVP
# =============================================================================


@jax.custom_jvp
def nufft2d3_jvp(
    x: Array,
    y: Array,
    c: Array,
    s: Array,
    t: Array,
    n_modes: tuple[int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """2D Type 3 NUFFT with forward-mode AD support."""
    return _nufft2d3_impl(x, y, c, s, t, n_modes, eps, isign, upsampfac)


@nufft2d3_jvp.defjvp
def _nufft2d3_jvp_rule(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """Forward-mode differentiation for nufft2d3."""
    x, y, c, s, t, n_modes, eps, isign, upsampfac = primals
    dx, dy, dc, ds, dt, _, _, _, _ = tangents

    # Primal output
    f = _nufft2d3_impl(x, y, c, s, t, n_modes, eps, isign, upsampfac)

    # Contribution from dc
    df_from_c = _nufft2d3_impl(x, y, dc, s, t, n_modes, eps, isign, upsampfac) if dc is not None else jnp.zeros_like(f)

    # Contribution from dx: s-weighted
    if dx is not None:
        weighted_transform = _nufft2d3_impl(x, y, c * dx, s, t, n_modes, eps, isign, upsampfac)
        df_from_x = 1j * isign * s * weighted_transform
    else:
        df_from_x = jnp.zeros_like(f)

    # Contribution from dy: t-weighted
    if dy is not None:
        weighted_transform = _nufft2d3_impl(x, y, c * dy, s, t, n_modes, eps, isign, upsampfac)
        df_from_y = 1j * isign * t * weighted_transform
    else:
        df_from_y = jnp.zeros_like(f)

    # Contribution from ds: x-weighted
    if ds is not None:
        x_weighted = _nufft2d3_impl(x, y, x * c, s, t, n_modes, eps, isign, upsampfac)
        df_from_s = 1j * isign * ds * x_weighted
    else:
        df_from_s = jnp.zeros_like(f)

    # Contribution from dt: y-weighted
    if dt is not None:
        y_weighted = _nufft2d3_impl(x, y, y * c, s, t, n_modes, eps, isign, upsampfac)
        df_from_t = 1j * isign * dt * y_weighted
    else:
        df_from_t = jnp.zeros_like(f)

    df = df_from_c + df_from_x + df_from_y + df_from_s + df_from_t

    return f, df


# =============================================================================
# 3D Type 3 NUFFT with custom JVP
# =============================================================================


@jax.custom_jvp
def nufft3d3_jvp(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    s: Array,
    t: Array,
    u: Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    """3D Type 3 NUFFT with forward-mode AD support."""
    return _nufft3d3_impl(x, y, z, c, s, t, u, n_modes, eps, isign, upsampfac)


@nufft3d3_jvp.defjvp
def _nufft3d3_jvp_rule(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    """Forward-mode differentiation for nufft3d3."""
    x, y, z, c, s, t, u, n_modes, eps, isign, upsampfac = primals
    dx, dy, dz, dc, ds, dt, du, _, _, _, _ = tangents

    # Primal output
    f = _nufft3d3_impl(x, y, z, c, s, t, u, n_modes, eps, isign, upsampfac)

    # Contribution from dc
    df_from_c = (
        _nufft3d3_impl(x, y, z, dc, s, t, u, n_modes, eps, isign, upsampfac) if dc is not None else jnp.zeros_like(f)
    )

    # Contribution from dx: s-weighted
    if dx is not None:
        weighted_transform = _nufft3d3_impl(x, y, z, c * dx, s, t, u, n_modes, eps, isign, upsampfac)
        df_from_x = 1j * isign * s * weighted_transform
    else:
        df_from_x = jnp.zeros_like(f)

    # Contribution from dy: t-weighted
    if dy is not None:
        weighted_transform = _nufft3d3_impl(x, y, z, c * dy, s, t, u, n_modes, eps, isign, upsampfac)
        df_from_y = 1j * isign * t * weighted_transform
    else:
        df_from_y = jnp.zeros_like(f)

    # Contribution from dz: u-weighted
    if dz is not None:
        weighted_transform = _nufft3d3_impl(x, y, z, c * dz, s, t, u, n_modes, eps, isign, upsampfac)
        df_from_z = 1j * isign * u * weighted_transform
    else:
        df_from_z = jnp.zeros_like(f)

    # Contribution from ds: x-weighted
    if ds is not None:
        x_weighted = _nufft3d3_impl(x, y, z, x * c, s, t, u, n_modes, eps, isign, upsampfac)
        df_from_s = 1j * isign * ds * x_weighted
    else:
        df_from_s = jnp.zeros_like(f)

    # Contribution from dt: y-weighted
    if dt is not None:
        y_weighted = _nufft3d3_impl(x, y, z, y * c, s, t, u, n_modes, eps, isign, upsampfac)
        df_from_t = 1j * isign * dt * y_weighted
    else:
        df_from_t = jnp.zeros_like(f)

    # Contribution from du: z-weighted
    if du is not None:
        z_weighted = _nufft3d3_impl(x, y, z, z * c, s, t, u, n_modes, eps, isign, upsampfac)
        df_from_u = 1j * isign * du * z_weighted
    else:
        df_from_u = jnp.zeros_like(f)

    df = df_from_c + df_from_x + df_from_y + df_from_z + df_from_s + df_from_t + df_from_u

    return f, df


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # 1D transforms with VJP
    "nufft1d1",
    "nufft1d2",
    "nufft1d3",
    # 1D transforms with JVP
    "nufft1d1_jvp",
    "nufft1d2_jvp",
    "nufft1d3_jvp",
    # 2D transforms with VJP
    "nufft2d1",
    "nufft2d2",
    "nufft2d3",
    # 2D transforms with JVP
    "nufft2d1_jvp",
    "nufft2d2_jvp",
    "nufft2d3_jvp",
    # 3D transforms with VJP
    "nufft3d1",
    "nufft3d2",
    "nufft3d3",
    # 3D transforms with JVP
    "nufft3d1_jvp",
    "nufft3d2_jvp",
    "nufft3d3_jvp",
]
