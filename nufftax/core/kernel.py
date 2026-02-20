"""
Kernel functions for NUFFT spreading/interpolation.

The kernel determines the accuracy of the NUFFT approximation.
Default is the ES (Exponential of Semicircle) kernel.

Reference: FINUFFT include/finufft_common/kernel.h
"""

import math
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
    # Cast all constants to input dtype to prevent float64 promotion
    dtype = z.dtype
    beta = jnp.array(beta, dtype=dtype)
    c = jnp.array(c, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)

    # Compute argument under sqrt
    arg = one - c * z * z

    # Mask for valid domain (arg >= 0)
    valid = arg >= zero

    # Compute kernel where valid
    sqrt_arg = jnp.sqrt(jnp.maximum(arg, zero))
    result = jnp.exp(beta * (sqrt_arg - one))

    # Zero outside support
    return jnp.where(valid, result, zero)


def es_kernel_with_derivative(
    z: jax.Array,
    beta: float,
    c: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute ES kernel and its derivative in a single pass.

    This is more efficient than computing them separately as it shares
    the sqrt and exp computations.

    phi(z) = exp(beta * (sqrt(1 - c*z²) - 1))
    dphi/dz = -beta * c * z / sqrt(1 - c*z²) * phi(z)

    Args:
        z: Points in kernel support, shape (...)
        beta: Shape parameter
        c: Normalization parameter

    Returns:
        (phi, dphi): Tuple of kernel values and derivatives
    """
    # Cast all constants to input dtype to prevent float64 promotion
    dtype = z.dtype
    beta = jnp.array(beta, dtype=dtype)
    c = jnp.array(c, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)
    eps = jnp.array(1e-14, dtype=dtype)

    # Compute argument under sqrt
    arg = one - c * z * z

    # Mask for valid domain
    valid = arg > eps  # Slightly positive to avoid division issues
    valid_for_phi = arg >= zero

    # Compute shared intermediates
    sqrt_arg = jnp.sqrt(jnp.maximum(arg, eps))

    # Compute kernel value
    phi = jnp.exp(beta * (sqrt_arg - one))
    phi = jnp.where(valid_for_phi, phi, zero)

    # Compute derivative: dphi/dz = -beta * c * z / sqrt_arg * phi
    dphi = -beta * c * z / sqrt_arg * phi
    dphi = jnp.where(valid, dphi, zero)

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


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def kernel_fourier_series(
    nf: int,
    nspread: int,
    beta: float,
    c: float,
    dtype: jnp.dtype | None = None,
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
        dtype: Output dtype (default: float32 if x64 disabled, else float64)

    Returns:
        phihat: Fourier coefficients, shape (nf//2 + 1,)
    """
    # Default dtype
    if dtype is None:
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    # Number of quadrature points - use enough for accuracy
    J2 = nspread / 2.0
    n_quad = max(int(4 + 3.0 * J2), 20)

    # Uniform quadrature over full kernel support [-J/2, J/2]
    z = jnp.linspace(-J2, J2, n_quad, dtype=dtype)
    dz = z[1] - z[0]  # Grid spacing

    # Evaluate kernel at quadrature points
    phi_vals = es_kernel(z, beta, c)

    # Compute Fourier coefficients using trapezoidal rule
    # phihat[k] = integral phi(z) * cos(2*pi*k*z/nf) dz
    k = jnp.arange(nf // 2 + 1, dtype=dtype)
    phase = jnp.array(2.0 * jnp.pi, dtype=dtype) * jnp.outer(k, z) / nf

    # Trapezoidal integration weights
    weights = jnp.ones(n_quad, dtype=dtype) * dz
    weights = weights.at[0].set(dz / 2)
    weights = weights.at[-1].set(dz / 2)

    phihat = jnp.sum(phi_vals[None, :] * weights[None, :] * jnp.cos(phase), axis=1)

    return phihat
