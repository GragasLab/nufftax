"""
Type 2 NUFFT transforms: Uniform to Nonuniform.

Type 2 evaluates Fourier series at nonuniform points:
    c[j] = sum_k f[k] * exp(isign * i * k * x[j])

Pipeline: deconvolve -> iFFT -> interpolate

Reference: FINUFFT finufft_core.cpp
"""

from functools import partial

import jax
import jax.numpy as jnp

from ..core.deconvolve import deconvolve_pad_1d, deconvolve_pad_2d, deconvolve_pad_3d
from ..core.kernel import compute_kernel_params, kernel_fourier_series
from ..core.spread import interp_1d, interp_2d, interp_3d
from ..utils.grid import compute_grid_size


@partial(jax.jit, static_argnames=("eps", "isign", "upsampfac", "modeord"))
def nufft1d2(
    x: jax.Array,
    f: jax.Array,
    eps: float = 1e-6,
    isign: int = -1,
    upsampfac: float = 2.0,
    modeord: int = 0,
) -> jax.Array:
    """
    1D Type-2 NUFFT: uniform to nonuniform.

    Computes:
        c[j] = sum_{k in K} f[k] * exp(isign * i * k * x[j])

    where K is the set of Fourier modes from -n_modes//2 to (n_modes-1)//2.

    Args:
        x: Nonuniform points in [-pi, pi) or [0, 2*pi), shape (M,)
        f: Fourier coefficients, shape (n_modes,) or (n_trans, n_modes)
        eps: Requested precision (tolerance)
        isign: Sign in the exponential (+1 or -1)
        upsampfac: Oversampling factor
        modeord: Mode ordering (0=CMCL, 1=FFT-style)

    Returns:
        c: Values at nonuniform points, shape (M,) or (n_trans, M)
    """
    # Handle batched input
    batched = f.ndim == 2
    if not batched:
        f = f[None, :]

    n_trans, n_modes = f.shape

    # Compute kernel parameters based on tolerance
    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    # Compute fine grid size
    nf = compute_grid_size(n_modes, upsampfac, nspread)

    # Compute kernel Fourier series coefficients
    phihat = kernel_fourier_series(nf, nspread, kernel_params.beta, kernel_params.c)

    # Step 1: Deconvolve and pad to fine grid
    fw_hat = deconvolve_pad_1d(f, phihat, nf, modeord)

    # Step 2: Inverse FFT
    # The sign convention: isign < 0 means exp(-ikx), which corresponds to IFFT * nf
    # isign > 0 means exp(+ikx), which corresponds to FFT
    fw = jnp.fft.ifft(fw_hat, axis=-1) * nf if isign < 0 else jnp.fft.fft(fw_hat, axis=-1)

    # Step 3: Normalize x to [-pi, pi) for the interp function
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Step 4: Interpolate to nonuniform points
    c = interp_1d(x_normalized, fw, nf, kernel_params)

    if not batched:
        c = c[0]

    return c


@partial(jax.jit, static_argnames=("eps", "isign", "upsampfac", "modeord"))
def nufft2d2(
    x: jax.Array,
    y: jax.Array,
    f: jax.Array,
    eps: float = 1e-6,
    isign: int = -1,
    upsampfac: float = 2.0,
    modeord: int = 0,
) -> jax.Array:
    """
    2D Type-2 NUFFT: uniform to nonuniform.

    Computes:
        c[j] = sum_{k1, k2} f[k1, k2] * exp(isign * i * (k1*x[j] + k2*y[j]))

    Args:
        x, y: Nonuniform points in [-pi, pi) or [0, 2*pi), shape (M,)
        f: Fourier coefficients, shape (n_modes2, n_modes1) or (n_trans, n_modes2, n_modes1)
        eps: Requested precision
        isign: Sign in the exponential
        upsampfac: Oversampling factor
        modeord: Mode ordering

    Returns:
        c: Values at nonuniform points, shape (M,) or (n_trans, M)
    """
    # Handle batched input
    batched = f.ndim == 3
    if not batched:
        f = f[None, :, :]

    n_trans, n_modes2, n_modes1 = f.shape

    # Compute kernel parameters
    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    # Compute fine grid sizes
    nf1 = compute_grid_size(n_modes1, upsampfac, nspread)
    nf2 = compute_grid_size(n_modes2, upsampfac, nspread)

    # Compute kernel Fourier series for each dimension
    phihat1 = kernel_fourier_series(nf1, nspread, kernel_params.beta, kernel_params.c)
    phihat2 = kernel_fourier_series(nf2, nspread, kernel_params.beta, kernel_params.c)

    # Step 1: Deconvolve and pad to fine grid
    fw_hat = deconvolve_pad_2d(f, phihat1, phihat2, nf1, nf2, modeord)

    # Step 2: Inverse 2D FFT
    fw = jnp.fft.ifft2(fw_hat, axes=(-2, -1)) * (nf1 * nf2) if isign < 0 else jnp.fft.fft2(fw_hat, axes=(-2, -1))

    # Step 3: Normalize coordinates to [-pi, pi)
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    y_normalized = jnp.mod(y + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Step 4: Interpolate to nonuniform points
    c = interp_2d(x_normalized, y_normalized, fw, nf1, nf2, kernel_params)

    if not batched:
        c = c[0]

    return c


@partial(jax.jit, static_argnames=("eps", "isign", "upsampfac", "modeord"))
def nufft3d2(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    f: jax.Array,
    eps: float = 1e-6,
    isign: int = -1,
    upsampfac: float = 2.0,
    modeord: int = 0,
) -> jax.Array:
    """
    3D Type-2 NUFFT: uniform to nonuniform.

    Computes:
        c[j] = sum_{k1, k2, k3} f[k1, k2, k3] * exp(isign * i * (k1*x + k2*y + k3*z))

    Args:
        x, y, z: Nonuniform points in [-pi, pi) or [0, 2*pi), shape (M,)
        f: Fourier coefficients, shape (n_modes3, n_modes2, n_modes1) or
           (n_trans, n_modes3, n_modes2, n_modes1)
        eps: Requested precision
        isign: Sign in the exponential
        upsampfac: Oversampling factor
        modeord: Mode ordering

    Returns:
        c: Values at nonuniform points, shape (M,) or (n_trans, M)
    """
    # Handle batched input
    batched = f.ndim == 4
    if not batched:
        f = f[None, :, :, :]

    n_trans, n_modes3, n_modes2, n_modes1 = f.shape

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

    # Step 1: Deconvolve and pad to fine grid
    fw_hat = deconvolve_pad_3d(f, phihat1, phihat2, phihat3, nf1, nf2, nf3, modeord)

    # Step 2: Inverse 3D FFT
    if isign < 0:
        fw = jnp.fft.ifftn(fw_hat, axes=(-3, -2, -1)) * (nf1 * nf2 * nf3)
    else:
        fw = jnp.fft.fftn(fw_hat, axes=(-3, -2, -1))

    # Step 3: Normalize coordinates to [-pi, pi)
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    y_normalized = jnp.mod(y + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    z_normalized = jnp.mod(z + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Step 4: Interpolate to nonuniform points
    c = interp_3d(x_normalized, y_normalized, z_normalized, fw, nf1, nf2, nf3, kernel_params)

    if not batched:
        c = c[0]

    return c
