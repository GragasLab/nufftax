"""
Type 1 NUFFT transforms: Nonuniform to Uniform.

Type 1 computes the Fourier coefficients from nonuniform point data:
    f[k] = sum_j c[j] * exp(isign * i * k * x[j])

Pipeline: spread -> FFT -> deconvolve

Reference: FINUFFT finufft_core.cpp
"""

import jax
import jax.numpy as jnp

from ..core.deconvolve import deconvolve_shuffle_1d, deconvolve_shuffle_2d, deconvolve_shuffle_3d
from ..core.kernel import compute_kernel_params, kernel_fourier_series
from ..core.spread import spread_1d, spread_2d, spread_3d
from ..utils.grid import compute_grid_size


def nufft1d1(
    x: jax.Array,
    c: jax.Array,
    n_modes: int,
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
    modeord: int = 0,
) -> jax.Array:
    """
    1D Type-1 NUFFT: nonuniform to uniform.

    Computes:
        f[k] = sum_{j=0}^{M-1} c[j] * exp(isign * i * k * x[j])

    for k = -n_modes//2, ..., (n_modes-1)//2 (CMCL ordering, modeord=0)
    or k = 0, ..., n_modes//2-1, -n_modes//2, ..., -1 (FFT ordering, modeord=1)

    Args:
        x: Nonuniform points in [-pi, pi) or [0, 2*pi), shape (M,)
        c: Complex strengths at nonuniform points, shape (M,) or (n_trans, M)
        n_modes: Number of output Fourier modes
        eps: Requested precision (tolerance), e.g., 1e-6
        isign: Sign in the exponential (+1 or -1)
        upsampfac: Oversampling factor (default 2.0)
        modeord: Mode ordering (0=CMCL, 1=FFT-style)

    Returns:
        f: Fourier coefficients, shape (n_modes,) or (n_trans, n_modes)
    """
    # Handle batched input
    batched = c.ndim == 2
    if not batched:
        c = c[None, :]

    n_trans, M = c.shape

    # Compute kernel parameters based on tolerance
    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    # Compute fine grid size
    nf = compute_grid_size(n_modes, upsampfac, nspread)

    # Compute kernel Fourier series coefficients for deconvolution
    phihat = kernel_fourier_series(nf, nspread, kernel_params.beta, kernel_params.c)

    # Step 1: Normalize x to [-pi, pi) for the spread function
    # If x is in [0, 2*pi), shift to [-pi, pi)
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Step 2: Spread to fine grid
    # The spread function expects coordinates in [-pi, pi)
    fw = spread_1d(x_normalized, c, nf, kernel_params)

    # Step 3: FFT
    # The sign convention: isign > 0 means exp(+ikx), which corresponds to FFT
    # isign < 0 means exp(-ikx), which corresponds to IFFT * nf
    fw_hat = jnp.fft.fft(fw, axis=-1) if isign > 0 else jnp.fft.ifft(fw, axis=-1) * nf

    # Step 4: Deconvolve and shuffle to output mode ordering
    f = deconvolve_shuffle_1d(fw_hat, phihat, n_modes, modeord)

    if not batched:
        f = f[0]

    return f


def nufft2d1(
    x: jax.Array,
    y: jax.Array,
    c: jax.Array,
    n_modes: int | tuple[int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
    modeord: int = 0,
) -> jax.Array:
    """
    2D Type-1 NUFFT: nonuniform to uniform.

    Computes:
        f[k1, k2] = sum_j c[j] * exp(isign * i * (k1*x[j] + k2*y[j]))

    Args:
        x, y: Nonuniform points in [-pi, pi) or [0, 2*pi), shape (M,)
        c: Complex strengths, shape (M,) or (n_trans, M)
        n_modes: Number of output modes (n_modes1, n_modes2) or single int for both
        eps: Requested precision
        isign: Sign in the exponential
        upsampfac: Oversampling factor
        modeord: Mode ordering

    Returns:
        f: Fourier coefficients, shape (n_modes2, n_modes1) or (n_trans, n_modes2, n_modes1)
    """
    # Handle n_modes
    if isinstance(n_modes, int):
        n_modes1 = n_modes2 = n_modes
    else:
        n_modes1, n_modes2 = n_modes

    # Handle batched input
    batched = c.ndim == 2
    if not batched:
        c = c[None, :]

    n_trans, M = c.shape

    # Compute kernel parameters
    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    # Compute fine grid sizes
    nf1 = compute_grid_size(n_modes1, upsampfac, nspread)
    nf2 = compute_grid_size(n_modes2, upsampfac, nspread)

    # Compute kernel Fourier series for each dimension
    phihat1 = kernel_fourier_series(nf1, nspread, kernel_params.beta, kernel_params.c)
    phihat2 = kernel_fourier_series(nf2, nspread, kernel_params.beta, kernel_params.c)

    # Normalize coordinates to [-pi, pi)
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    y_normalized = jnp.mod(y + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Spread to fine grid
    fw = spread_2d(x_normalized, y_normalized, c, nf1, nf2, kernel_params)

    # 2D FFT
    fw_hat = jnp.fft.fft2(fw, axes=(-2, -1)) if isign > 0 else jnp.fft.ifft2(fw, axes=(-2, -1)) * (nf1 * nf2)

    # Deconvolve and shuffle
    f = deconvolve_shuffle_2d(fw_hat, phihat1, phihat2, n_modes1, n_modes2, modeord)

    if not batched:
        f = f[0]

    return f


def nufft3d1(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    c: jax.Array,
    n_modes: int | tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
    modeord: int = 0,
) -> jax.Array:
    """
    3D Type-1 NUFFT: nonuniform to uniform.

    Computes:
        f[k1, k2, k3] = sum_j c[j] * exp(isign * i * (k1*x[j] + k2*y[j] + k3*z[j]))

    Args:
        x, y, z: Nonuniform points in [-pi, pi) or [0, 2*pi), shape (M,)
        c: Complex strengths, shape (M,) or (n_trans, M)
        n_modes: Number of output modes (n_modes1, n_modes2, n_modes3) or single int
        eps: Requested precision
        isign: Sign in the exponential
        upsampfac: Oversampling factor
        modeord: Mode ordering

    Returns:
        f: Fourier coefficients, shape (n_modes3, n_modes2, n_modes1) or
           (n_trans, n_modes3, n_modes2, n_modes1)
    """
    # Handle n_modes
    if isinstance(n_modes, int):
        n_modes1 = n_modes2 = n_modes3 = n_modes
    else:
        n_modes1, n_modes2, n_modes3 = n_modes

    # Handle batched input
    batched = c.ndim == 2
    if not batched:
        c = c[None, :]

    n_trans, M = c.shape

    # Compute kernel parameters
    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    # Compute fine grid sizes
    nf1 = compute_grid_size(n_modes1, upsampfac, nspread)
    nf2 = compute_grid_size(n_modes2, upsampfac, nspread)
    nf3 = compute_grid_size(n_modes3, upsampfac, nspread)

    # Compute kernel Fourier series for each dimension
    phihat1 = kernel_fourier_series(nf1, nspread, kernel_params.beta, kernel_params.c)
    phihat2 = kernel_fourier_series(nf2, nspread, kernel_params.beta, kernel_params.c)
    phihat3 = kernel_fourier_series(nf3, nspread, kernel_params.beta, kernel_params.c)

    # Normalize coordinates to [-pi, pi)
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    y_normalized = jnp.mod(y + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    z_normalized = jnp.mod(z + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Spread to fine grid
    fw = spread_3d(x_normalized, y_normalized, z_normalized, c, nf1, nf2, nf3, kernel_params)

    # 3D FFT
    if isign > 0:
        fw_hat = jnp.fft.fftn(fw, axes=(-3, -2, -1))
    else:
        fw_hat = jnp.fft.ifftn(fw, axes=(-3, -2, -1)) * (nf1 * nf2 * nf3)

    # Deconvolve and shuffle
    f = deconvolve_shuffle_3d(fw_hat, phihat1, phihat2, phihat3, n_modes1, n_modes2, n_modes3, modeord)

    if not batched:
        f = f[0]

    return f
