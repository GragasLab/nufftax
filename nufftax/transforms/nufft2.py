"""
Type 2 NUFFT transforms: Uniform to Nonuniform.

Type 2 evaluates Fourier series at nonuniform points:
    c[j] = sum_k f[k] * exp(isign * i * k * x[j])

Pipeline: deconvolve -> iFFT -> interpolate

Reference: FINUFFT finufft_core.cpp
"""

import jax
import jax.numpy as jnp

from ..core.deconvolve import deconvolve_pad_1d_vectorized
from ..core.kernel import compute_kernel_params, kernel_fourier_series
from ..core.spread import interp_1d, interp_2d, interp_3d
from ..utils.grid import compute_grid_size


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
    x.shape[0]

    # Compute kernel parameters based on tolerance
    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    # Compute fine grid size
    nf = compute_grid_size(n_modes, upsampfac, nspread)

    # Compute kernel Fourier series coefficients
    phihat = kernel_fourier_series(nf, nspread, kernel_params.beta, kernel_params.c)

    # Step 1: Deconvolve and pad to fine grid
    fw_hat = deconvolve_pad_1d_vectorized(f, phihat, nf, modeord)

    # Step 2: Inverse FFT
    # The sign convention: isign < 0 means exp(-ikx), which corresponds to IFFT * nf
    # isign > 0 means exp(+ikx), which corresponds to FFT
    fw = jnp.fft.ifft(fw_hat, axis=-1) * nf if isign < 0 else jnp.fft.fft(fw_hat, axis=-1)

    # Step 3: Normalize x to [-pi, pi) for the interp function
    x_normalized = jnp.mod(x + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Step 4: Interpolate to nonuniform points
    # Note: interp functions don't need nf passed separately, they get it from fw shape
    c = interp_1d(x_normalized, fw, nf, kernel_params)

    if not batched:
        c = c[0]

    return c


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
    x.shape[0]

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
    fw_hat = _deconvolve_pad_2d_vectorized(f, phihat1, phihat2, nf1, nf2, modeord)

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
    x.shape[0]

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
    fw_hat = _deconvolve_pad_3d_vectorized(f, phihat1, phihat2, phihat3, nf1, nf2, nf3, modeord)

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


# ============================================================================
# Helper functions for 2D and 3D deconvolution/padding (vectorized)
# ============================================================================


def _build_deconv_factors_1d(phihat, n_modes, modeord=0):
    """Build 1D deconvolution factors in mode order."""
    kmin = -n_modes // 2
    kmax = (n_modes - 1) // 2

    factors_pos = 1.0 / phihat[: kmax + 1]
    factors_neg = 1.0 / phihat[1 : -kmin + 1][::-1]

    if modeord == 0:
        # CMCL: negative first, then positive
        factors = jnp.concatenate([factors_neg, factors_pos])
    else:
        factors = jnp.concatenate([factors_pos, factors_neg])

    return factors


def _deconvolve_pad_2d_vectorized(
    f: jax.Array,
    phihat1: jax.Array,
    phihat2: jax.Array,
    nf1: int,
    nf2: int,
    modeord: int = 0,
) -> jax.Array:
    """Vectorized 2D deconvolution and padding for Type 2."""
    batched = f.ndim == 3
    if not batched:
        f = f[None, :, :]

    n_trans, n_modes2, n_modes1 = f.shape

    k1_min = -n_modes1 // 2
    k1_max = (n_modes1 - 1) // 2
    k2_min = -n_modes2 // 2
    k2_max = (n_modes2 - 1) // 2

    # Build 2D deconvolution factors
    factors_1d_x = _build_deconv_factors_1d(phihat1, n_modes1, modeord)
    factors_1d_y = _build_deconv_factors_1d(phihat2, n_modes2, modeord)
    factors_2d = factors_1d_y[:, None] * factors_1d_x[None, :]

    # Deconvolve
    f_deconv = f * factors_2d

    # Split into positive and negative frequency components
    if modeord == 0:
        n_neg_x = -k1_min
        n_neg_y = -k2_min
        # Split y: first n_neg_y rows are negative, rest are positive
        f_negy_negx = f_deconv[:, :n_neg_y, :n_neg_x]
        f_negy_posx = f_deconv[:, :n_neg_y, n_neg_x:]
        f_posy_negx = f_deconv[:, n_neg_y:, :n_neg_x]
        f_posy_posx = f_deconv[:, n_neg_y:, n_neg_x:]
    else:
        n_pos_x = k1_max + 1
        n_pos_y = k2_max + 1
        f_posy_posx = f_deconv[:, :n_pos_y, :n_pos_x]
        f_posy_negx = f_deconv[:, :n_pos_y, n_pos_x:]
        f_negy_posx = f_deconv[:, n_pos_y:, :n_pos_x]
        f_negy_negx = f_deconv[:, n_pos_y:, n_pos_x:]

    # Create output and place components
    fw_hat = jnp.zeros((n_trans, nf2, nf1), dtype=f.dtype)

    # Positive y, positive x: at (0:k2_max+1, 0:k1_max+1)
    fw_hat = fw_hat.at[:, : k2_max + 1, : k1_max + 1].set(f_posy_posx)

    # Positive y, negative x: at (0:k2_max+1, nf1+k1_min:nf1)
    fw_hat = fw_hat.at[:, : k2_max + 1, nf1 + k1_min :].set(f_posy_negx)

    # Negative y, positive x: at (nf2+k2_min:nf2, 0:k1_max+1)
    fw_hat = fw_hat.at[:, nf2 + k2_min :, : k1_max + 1].set(f_negy_posx)

    # Negative y, negative x: at (nf2+k2_min:nf2, nf1+k1_min:nf1)
    fw_hat = fw_hat.at[:, nf2 + k2_min :, nf1 + k1_min :].set(f_negy_negx)

    if not batched:
        fw_hat = fw_hat[0]

    return fw_hat


def _deconvolve_pad_3d_vectorized(
    f: jax.Array,
    phihat1: jax.Array,
    phihat2: jax.Array,
    phihat3: jax.Array,
    nf1: int,
    nf2: int,
    nf3: int,
    modeord: int = 0,
) -> jax.Array:
    """Vectorized 3D deconvolution and padding for Type 2."""
    batched = f.ndim == 4
    if not batched:
        f = f[None, :, :, :]

    n_trans, n_modes3, n_modes2, n_modes1 = f.shape

    k1_min, k1_max = -n_modes1 // 2, (n_modes1 - 1) // 2
    k2_min, k2_max = -n_modes2 // 2, (n_modes2 - 1) // 2
    k3_min, k3_max = -n_modes3 // 2, (n_modes3 - 1) // 2

    # Build 3D deconvolution factors
    factors_1d_x = _build_deconv_factors_1d(phihat1, n_modes1, modeord)
    factors_1d_y = _build_deconv_factors_1d(phihat2, n_modes2, modeord)
    factors_1d_z = _build_deconv_factors_1d(phihat3, n_modes3, modeord)
    factors_3d = factors_1d_z[:, None, None] * factors_1d_y[None, :, None] * factors_1d_x[None, None, :]

    # Deconvolve
    f_deconv = f * factors_3d

    # Split based on mode ordering
    if modeord == 0:
        n_neg_x, n_neg_y, n_neg_z = -k1_min, -k2_min, -k3_min
        # Extract 8 octants
        # z-negative
        f_nz_ny_nx = f_deconv[:, :n_neg_z, :n_neg_y, :n_neg_x]
        f_nz_ny_px = f_deconv[:, :n_neg_z, :n_neg_y, n_neg_x:]
        f_nz_py_nx = f_deconv[:, :n_neg_z, n_neg_y:, :n_neg_x]
        f_nz_py_px = f_deconv[:, :n_neg_z, n_neg_y:, n_neg_x:]
        # z-positive
        f_pz_ny_nx = f_deconv[:, n_neg_z:, :n_neg_y, :n_neg_x]
        f_pz_ny_px = f_deconv[:, n_neg_z:, :n_neg_y, n_neg_x:]
        f_pz_py_nx = f_deconv[:, n_neg_z:, n_neg_y:, :n_neg_x]
        f_pz_py_px = f_deconv[:, n_neg_z:, n_neg_y:, n_neg_x:]
    else:
        n_pos_x, n_pos_y, n_pos_z = k1_max + 1, k2_max + 1, k3_max + 1
        # Extract 8 octants
        # z-positive
        f_pz_py_px = f_deconv[:, :n_pos_z, :n_pos_y, :n_pos_x]
        f_pz_py_nx = f_deconv[:, :n_pos_z, :n_pos_y, n_pos_x:]
        f_pz_ny_px = f_deconv[:, :n_pos_z, n_pos_y:, :n_pos_x]
        f_pz_ny_nx = f_deconv[:, :n_pos_z, n_pos_y:, n_pos_x:]
        # z-negative
        f_nz_py_px = f_deconv[:, n_pos_z:, :n_pos_y, :n_pos_x]
        f_nz_py_nx = f_deconv[:, n_pos_z:, :n_pos_y, n_pos_x:]
        f_nz_ny_px = f_deconv[:, n_pos_z:, n_pos_y:, :n_pos_x]
        f_nz_ny_nx = f_deconv[:, n_pos_z:, n_pos_y:, n_pos_x:]

    # Create output and place 8 octants
    fw_hat = jnp.zeros((n_trans, nf3, nf2, nf1), dtype=f.dtype)

    # pz, py, px: (0:k3_max+1, 0:k2_max+1, 0:k1_max+1)
    fw_hat = fw_hat.at[:, : k3_max + 1, : k2_max + 1, : k1_max + 1].set(f_pz_py_px)

    # pz, py, nx: (0:k3_max+1, 0:k2_max+1, nf1+k1_min:nf1)
    fw_hat = fw_hat.at[:, : k3_max + 1, : k2_max + 1, nf1 + k1_min :].set(f_pz_py_nx)

    # pz, ny, px: (0:k3_max+1, nf2+k2_min:nf2, 0:k1_max+1)
    fw_hat = fw_hat.at[:, : k3_max + 1, nf2 + k2_min :, : k1_max + 1].set(f_pz_ny_px)

    # pz, ny, nx: (0:k3_max+1, nf2+k2_min:nf2, nf1+k1_min:nf1)
    fw_hat = fw_hat.at[:, : k3_max + 1, nf2 + k2_min :, nf1 + k1_min :].set(f_pz_ny_nx)

    # nz, py, px: (nf3+k3_min:nf3, 0:k2_max+1, 0:k1_max+1)
    fw_hat = fw_hat.at[:, nf3 + k3_min :, : k2_max + 1, : k1_max + 1].set(f_nz_py_px)

    # nz, py, nx: (nf3+k3_min:nf3, 0:k2_max+1, nf1+k1_min:nf1)
    fw_hat = fw_hat.at[:, nf3 + k3_min :, : k2_max + 1, nf1 + k1_min :].set(f_nz_py_nx)

    # nz, ny, px: (nf3+k3_min:nf3, nf2+k2_min:nf2, 0:k1_max+1)
    fw_hat = fw_hat.at[:, nf3 + k3_min :, nf2 + k2_min :, : k1_max + 1].set(f_nz_ny_px)

    # nz, ny, nx: (nf3+k3_min:nf3, nf2+k2_min:nf2, nf1+k1_min:nf1)
    fw_hat = fw_hat.at[:, nf3 + k3_min :, nf2 + k2_min :, nf1 + k1_min :].set(f_nz_ny_nx)

    if not batched:
        fw_hat = fw_hat[0]

    return fw_hat
