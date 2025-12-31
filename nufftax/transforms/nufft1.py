"""
Type 1 NUFFT transforms: Nonuniform to Uniform.

Type 1 computes the Fourier coefficients from nonuniform point data:
    f[k] = sum_j c[j] * exp(isign * i * k * x[j])

Pipeline: spread -> FFT -> deconvolve

Reference: FINUFFT finufft_core.cpp
"""

import jax
import jax.numpy as jnp

from ..core.deconvolve import deconvolve_shuffle_1d_vectorized
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
    f = deconvolve_shuffle_1d_vectorized(fw_hat, phihat, n_modes, modeord)

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

    # Deconvolve and shuffle (2D version)
    f = _deconvolve_shuffle_2d_vectorized(fw_hat, phihat1, phihat2, n_modes1, n_modes2, modeord)

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

    # Deconvolve and shuffle (3D version)
    f = _deconvolve_shuffle_3d_vectorized(fw_hat, phihat1, phihat2, phihat3, n_modes1, n_modes2, n_modes3, modeord)

    if not batched:
        f = f[0]

    return f


# ============================================================================
# Helper functions for 2D and 3D deconvolution (vectorized)
# ============================================================================


def _build_indices_2d(nf1: int, nf2: int, n_modes1: int, n_modes2: int, modeord: int = 0):
    """Build extraction indices for 2D deconvolution."""
    k1_min = -n_modes1 // 2
    k1_max = (n_modes1 - 1) // 2
    k2_min = -n_modes2 // 2
    k2_max = (n_modes2 - 1) // 2

    # Indices for x dimension
    idx_pos_x = jnp.arange(k1_max + 1)
    idx_neg_x = jnp.arange(nf1 + k1_min, nf1)

    # Indices for y dimension
    idx_pos_y = jnp.arange(k2_max + 1)
    idx_neg_y = jnp.arange(nf2 + k2_min, nf2)

    if modeord == 0:
        # CMCL: negative first, then positive for each dimension
        idx_x = jnp.concatenate([idx_neg_x, idx_pos_x])
        idx_y = jnp.concatenate([idx_neg_y, idx_pos_y])
    else:
        idx_x = jnp.concatenate([idx_pos_x, idx_neg_x])
        idx_y = jnp.concatenate([idx_pos_y, idx_neg_y])

    return idx_x, idx_y


def _build_deconv_factors_2d(phihat1, phihat2, n_modes1, n_modes2, modeord=0):
    """Build 2D deconvolution factors as outer product."""
    k1_min = -n_modes1 // 2
    k1_max = (n_modes1 - 1) // 2
    k2_min = -n_modes2 // 2
    k2_max = (n_modes2 - 1) // 2

    # 1D factors
    factors_pos_x = 1.0 / phihat1[: k1_max + 1]
    factors_neg_x = 1.0 / phihat1[1 : -k1_min + 1][::-1]
    factors_pos_y = 1.0 / phihat2[: k2_max + 1]
    factors_neg_y = 1.0 / phihat2[1 : -k2_min + 1][::-1]

    if modeord == 0:
        factors_x = jnp.concatenate([factors_neg_x, factors_pos_x])
        factors_y = jnp.concatenate([factors_neg_y, factors_pos_y])
    else:
        factors_x = jnp.concatenate([factors_pos_x, factors_neg_x])
        factors_y = jnp.concatenate([factors_pos_y, factors_neg_y])

    # 2D factors via outer product
    factors_2d = factors_y[:, None] * factors_x[None, :]

    return factors_2d


def _deconvolve_shuffle_2d_vectorized(
    fw_hat: jax.Array,
    phihat1: jax.Array,
    phihat2: jax.Array,
    n_modes1: int,
    n_modes2: int,
    modeord: int = 0,
) -> jax.Array:
    """Vectorized 2D deconvolution and shuffling."""
    nf1 = fw_hat.shape[-1]
    nf2 = fw_hat.shape[-2]

    # Build indices and factors
    idx_x, idx_y = _build_indices_2d(nf1, nf2, n_modes1, n_modes2, modeord)
    factors_2d = _build_deconv_factors_2d(phihat1, phihat2, n_modes1, n_modes2, modeord)

    # Extract using advanced indexing: fw_hat[..., idx_y, :][:, :, idx_x]
    # For batched: (n_trans, nf2, nf1) -> (n_trans, n_modes2, n_modes1)
    f = fw_hat[..., idx_y, :][..., idx_x] * factors_2d

    return f


def _build_indices_3d(nf1, nf2, nf3, n_modes1, n_modes2, n_modes3, modeord=0):
    """Build extraction indices for 3D deconvolution."""
    k1_min, k1_max = -n_modes1 // 2, (n_modes1 - 1) // 2
    k2_min, k2_max = -n_modes2 // 2, (n_modes2 - 1) // 2
    k3_min, k3_max = -n_modes3 // 2, (n_modes3 - 1) // 2

    idx_pos_x = jnp.arange(k1_max + 1)
    idx_neg_x = jnp.arange(nf1 + k1_min, nf1)
    idx_pos_y = jnp.arange(k2_max + 1)
    idx_neg_y = jnp.arange(nf2 + k2_min, nf2)
    idx_pos_z = jnp.arange(k3_max + 1)
    idx_neg_z = jnp.arange(nf3 + k3_min, nf3)

    if modeord == 0:
        idx_x = jnp.concatenate([idx_neg_x, idx_pos_x])
        idx_y = jnp.concatenate([idx_neg_y, idx_pos_y])
        idx_z = jnp.concatenate([idx_neg_z, idx_pos_z])
    else:
        idx_x = jnp.concatenate([idx_pos_x, idx_neg_x])
        idx_y = jnp.concatenate([idx_pos_y, idx_neg_y])
        idx_z = jnp.concatenate([idx_pos_z, idx_neg_z])

    return idx_x, idx_y, idx_z


def _build_deconv_factors_3d(phihat1, phihat2, phihat3, n_modes1, n_modes2, n_modes3, modeord=0):
    """Build 3D deconvolution factors."""
    k1_min, k1_max = -n_modes1 // 2, (n_modes1 - 1) // 2
    k2_min, k2_max = -n_modes2 // 2, (n_modes2 - 1) // 2
    k3_min, k3_max = -n_modes3 // 2, (n_modes3 - 1) // 2

    factors_pos_x = 1.0 / phihat1[: k1_max + 1]
    factors_neg_x = 1.0 / phihat1[1 : -k1_min + 1][::-1]
    factors_pos_y = 1.0 / phihat2[: k2_max + 1]
    factors_neg_y = 1.0 / phihat2[1 : -k2_min + 1][::-1]
    factors_pos_z = 1.0 / phihat3[: k3_max + 1]
    factors_neg_z = 1.0 / phihat3[1 : -k3_min + 1][::-1]

    if modeord == 0:
        factors_x = jnp.concatenate([factors_neg_x, factors_pos_x])
        factors_y = jnp.concatenate([factors_neg_y, factors_pos_y])
        factors_z = jnp.concatenate([factors_neg_z, factors_pos_z])
    else:
        factors_x = jnp.concatenate([factors_pos_x, factors_neg_x])
        factors_y = jnp.concatenate([factors_pos_y, factors_neg_y])
        factors_z = jnp.concatenate([factors_pos_z, factors_neg_z])

    # 3D factors via outer product
    factors_3d = factors_z[:, None, None] * factors_y[None, :, None] * factors_x[None, None, :]

    return factors_3d


def _deconvolve_shuffle_3d_vectorized(
    fw_hat: jax.Array,
    phihat1: jax.Array,
    phihat2: jax.Array,
    phihat3: jax.Array,
    n_modes1: int,
    n_modes2: int,
    n_modes3: int,
    modeord: int = 0,
) -> jax.Array:
    """Vectorized 3D deconvolution and shuffling."""
    nf1, nf2, nf3 = fw_hat.shape[-1], fw_hat.shape[-2], fw_hat.shape[-3]

    # Build indices and factors
    idx_x, idx_y, idx_z = _build_indices_3d(nf1, nf2, nf3, n_modes1, n_modes2, n_modes3, modeord)
    factors_3d = _build_deconv_factors_3d(phihat1, phihat2, phihat3, n_modes1, n_modes2, n_modes3, modeord)

    # Extract using advanced indexing
    # fw_hat shape: (n_trans, nf3, nf2, nf1) or (nf3, nf2, nf1)
    f = fw_hat[..., idx_z, :, :][..., idx_y, :][..., idx_x] * factors_3d

    return f
