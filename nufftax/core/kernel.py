"""
Kernel functions for NUFFT spreading/interpolation.

The kernel determines the accuracy of the NUFFT approximation.
Default is the ES (Exponential of Semicircle) kernel.

Reference: FINUFFT include/finufft_common/kernel.h
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class KernelParams(NamedTuple):
    """Parameters for the NUFFT spreading kernel."""

    nspread: int  # Kernel width (support in grid points)
    beta: float  # Shape parameter
    c: float  # Normalization parameter = 4/nspread^2
    upsampfac: float  # Upsampling factor (default 2.0)


def es_kernel(z: jax.Array, beta: float, c: float) -> jax.Array:
    """
    Evaluate the ES (Exponential of Semicircle) kernel.

    phi(z) = exp(beta * (sqrt(1 - c*z²) - 1))

    Args:
        z: Points in kernel support, shape (...)
        beta: Shape parameter (controls concentration)
        c: Parameter c = 4/nspread² for normalization

    Returns:
        Kernel values, same shape as z
    """
    # Compute argument under sqrt
    arg = 1.0 - c * z * z

    # Mask for valid domain (arg >= 0)
    valid = arg >= 0.0

    # Compute kernel where valid
    sqrt_arg = jnp.sqrt(jnp.maximum(arg, 0.0))
    result = jnp.exp(beta * (sqrt_arg - 1.0))

    # Zero outside support
    return jnp.where(valid, result, 0.0)


def es_kernel_derivative(z: jax.Array, beta: float, c: float) -> jax.Array:
    """
    Derivative of ES kernel with respect to z.

    d/dz phi(z) = -beta * c * z / sqrt(1 - c*z²) * phi(z)

    Args:
        z: Points in kernel support
        beta: Shape parameter
        c: Normalization parameter

    Returns:
        Kernel derivative values
    """
    arg = 1.0 - c * z * z
    valid = arg > 1e-14  # Avoid division by zero

    sqrt_arg = jnp.sqrt(jnp.maximum(arg, 1e-14))
    phi = jnp.exp(beta * (sqrt_arg - 1.0))
    dphi = -beta * c * z / sqrt_arg * phi

    return jnp.where(valid, dphi, 0.0)


def es_kernel_with_derivative(z: jax.Array, beta: float, c: float) -> tuple[jax.Array, jax.Array]:
    """
    Compute ES kernel and its derivative together (fused for efficiency).

    This avoids redundant computation of sqrt and exp when both
    kernel value and derivative are needed (e.g., in gradient computation).

    Args:
        z: Points in kernel support, shape (...)
        beta: Shape parameter
        c: Normalization parameter

    Returns:
        phi: Kernel values, same shape as z
        dphi: Kernel derivative values, same shape as z
    """
    arg = 1.0 - c * z * z
    valid = arg > 1e-14  # Avoid division by zero

    sqrt_arg = jnp.sqrt(jnp.maximum(arg, 1e-14))
    phi = jnp.exp(beta * (sqrt_arg - 1.0))
    dphi = -beta * c * z / sqrt_arg * phi

    # Zero outside support
    phi = jnp.where(arg >= 0.0, phi, 0.0)
    dphi = jnp.where(valid, dphi, 0.0)

    return phi, dphi


def compute_kernel_params(
    tol: float,
    upsampfac: float = 2.0,
    max_nspread: int = 16,
) -> KernelParams:
    """
    Compute kernel parameters for a given tolerance.

    This determines nspread (kernel width) and beta (shape parameter)
    to achieve the requested approximation accuracy.

    Args:
        tol: Requested precision (e.g., 1e-6)
        upsampfac: Oversampling factor (default 2.0)
        max_nspread: Maximum allowed kernel width

    Returns:
        KernelParams with nspread, beta, c, upsampfac
    """
    # Heuristic formula for nspread based on tolerance
    # nspread ≈ ceil(log10(1/tol)) + 1
    import math

    log_tol = -math.log10(max(tol, 1e-16))
    nspread = int(math.ceil(log_tol + 1))
    nspread = max(2, min(nspread, max_nspread))

    # Make nspread even for symmetry (optional)
    if nspread % 2 == 1:
        nspread += 1

    # Compute beta using empirical formula from FINUFFT
    # beta = pi * sqrt(nspread² / sigma² * (sigma - 0.5)² - 0.8)
    sigma = upsampfac
    inner = (nspread**2) / (sigma**2) * (sigma - 0.5) ** 2 - 0.8
    if inner > 0:
        beta = math.pi * math.sqrt(inner)
    else:
        beta = 2.3 * nspread  # Fallback

    # c = 4/nspread²
    c = 4.0 / (nspread**2)

    return KernelParams(nspread=nspread, beta=beta, c=c, upsampfac=upsampfac)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def kernel_fourier_series(
    nf: int,
    nspread: int,
    beta: float,
    c: float,
) -> jax.Array:
    """
    Compute Fourier series coefficients of the kernel.

    These are used in the deconvolution step to correct for spreading.
    Uses numerical quadrature on the Euler-Fourier formula.

    Args:
        nf: Fine grid size
        nspread: Kernel width
        beta: Shape parameter
        c: Normalization parameter

    Returns:
        phihat: Fourier coefficients, shape (nf//2 + 1,)
    """
    # Number of quadrature points
    J2 = nspread / 2.0
    n_quad = int(2 + 3.0 * J2)

    # Gauss-Legendre quadrature nodes and weights on [0, 1]
    # We'll use a simple approximation with uniform spacing for now
    # TODO: Use proper Gauss-Legendre quadrature
    t = jnp.linspace(0, 1, n_quad + 2)[1:-1]  # Exclude endpoints
    w = jnp.ones(n_quad) / n_quad

    # Map to [0, J/2]
    z = t * J2
    weights = w * J2

    # Evaluate kernel at quadrature points
    f = es_kernel(z, beta, c) * weights

    # Compute Fourier coefficients via phase winding
    k = jnp.arange(nf // 2 + 1)

    # phihat[k] = 2 * sum_n f[n] * cos(2*pi*k*z[n]/nf)
    # (factor of 2 for reflection symmetry)
    phase = 2.0 * jnp.pi * jnp.outer(k, z) / nf
    phihat = 2.0 * jnp.sum(f[None, :] * jnp.cos(phase), axis=1)

    return phihat


def evaluate_kernel_horner(
    z: jax.Array,
    coeffs: jax.Array,
    nspread: int,
) -> jax.Array:
    """
    Evaluate kernel using Horner polynomial approximation.

    This is faster than direct kernel evaluation for JIT compilation.

    Args:
        z: Points in [-1, 1] (normalized kernel support)
        coeffs: Horner coefficients, shape (nc, nspread)
        nspread: Kernel width

    Returns:
        Kernel values at z
    """
    # TODO: Implement piecewise Horner evaluation
    # For now, fall back to direct evaluation
    raise NotImplementedError("Horner evaluation not yet implemented")
