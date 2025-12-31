"""
Deconvolution functions for NUFFT.

The deconvolution step corrects for the spreading kernel by dividing
the Fourier coefficients by the kernel's Fourier transform.

For Type 1: deconvolve and shuffle from fine grid FFT to output modes
For Type 2: deconvolve and pad from input modes to fine grid for iFFT

All implementations are fully vectorized (no Python for loops) for
optimal JAX/XLA performance on GPU.

Reference: FINUFFT src/finufft_core.cpp (deconvolveshuffle1d, etc.)
"""

from functools import partial

import jax
import jax.numpy as jnp

# ============================================================================
# Helper functions for building indices and factors
# ============================================================================


def _build_deconv_factors_1d(phihat: jax.Array, n_modes: int, modeord: int = 0):
    """Build deconvolution factor array for 1D case."""
    kmin = -n_modes // 2
    kmax = (n_modes - 1) // 2

    # Build factor array in output mode order
    if modeord == 0:
        # CMCL: [kmin, ..., -1, 0, 1, ..., kmax]
        factors_neg = 1.0 / phihat[1 : -kmin + 1][::-1]
        factors_pos = 1.0 / phihat[: kmax + 1]
        factors = jnp.concatenate([factors_neg, factors_pos])
    else:
        # FFT-style: [0, ..., kmax, kmin, ..., -1]
        factors_pos = 1.0 / phihat[: kmax + 1]
        factors_neg = 1.0 / phihat[1 : -kmin + 1][::-1]
        factors = jnp.concatenate([factors_pos, factors_neg])

    return factors


def _build_extraction_indices_1d(nf: int, n_modes: int, modeord: int = 0):
    """Build indices to extract modes from FFT output."""
    kmin = -n_modes // 2
    kmax = (n_modes - 1) // 2

    # Indices in fw_hat for positive and negative frequencies
    idx_pos = jnp.arange(kmax + 1)
    idx_neg = jnp.arange(nf + kmin, nf)

    if modeord == 0:
        indices = jnp.concatenate([idx_neg, idx_pos])
    else:
        indices = jnp.concatenate([idx_pos, idx_neg])

    return indices


# ============================================================================
# 1D Deconvolution (Vectorized)
# ============================================================================


@partial(jax.jit, static_argnums=(2, 3))
def deconvolve_shuffle_1d(
    fw_hat: jax.Array,
    phihat: jax.Array,
    n_modes: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Deconvolve and shuffle 1D FFT output to Fourier coefficients (Type 1).

    Fully vectorized implementation.

    Args:
        fw_hat: FFT of fine grid, shape (nf,) or (n_trans, nf)
        phihat: Kernel Fourier coefficients, shape (nf//2 + 1,)
        n_modes: Number of output modes
        modeord: Mode ordering (0=CMCL, 1=FFT-style)

    Returns:
        f: Fourier coefficients, shape (n_modes,) or (n_trans, n_modes)
    """
    nf = fw_hat.shape[-1]

    # Build indices and factors
    indices = _build_extraction_indices_1d(nf, n_modes, modeord)
    factors = _build_deconv_factors_1d(phihat, n_modes, modeord)

    # Extract and deconvolve in one step
    f = fw_hat[..., indices] * factors

    return f


@partial(jax.jit, static_argnums=(2, 3))
def deconvolve_pad_1d(
    f: jax.Array,
    phihat: jax.Array,
    nf: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Deconvolve and pad 1D Fourier coefficients for iFFT (Type 2).

    Fully vectorized implementation.

    Args:
        f: Fourier coefficients, shape (n_modes,) or (n_trans, n_modes)
        phihat: Kernel Fourier coefficients, shape (nf//2 + 1,)
        nf: Fine grid size
        modeord: Mode ordering (0=CMCL, 1=FFT-style)

    Returns:
        fw_hat: Padded and deconvolved grid for iFFT, shape (nf,) or (n_trans, nf)
    """
    n_modes = f.shape[-1]
    kmin = -n_modes // 2
    kmax = (n_modes - 1) // 2

    # Build factors in input order
    factors = _build_deconv_factors_1d(phihat, n_modes, modeord)

    # Deconvolve
    f_deconv = f * factors

    # Split into positive and negative frequencies
    if modeord == 0:
        n_neg = -kmin
        f_neg = f_deconv[..., :n_neg]
        f_pos = f_deconv[..., n_neg:]
    else:
        n_pos = kmax + 1
        f_pos = f_deconv[..., :n_pos]
        f_neg = f_deconv[..., n_pos:]

    # Create output with zeros and place values
    if f.ndim == 1:
        fw_hat = jnp.zeros(nf, dtype=f.dtype)
        fw_hat = fw_hat.at[: kmax + 1].set(f_pos)
        fw_hat = fw_hat.at[nf + kmin :].set(f_neg)
    else:
        n_trans = f.shape[0]
        fw_hat = jnp.zeros((n_trans, nf), dtype=f.dtype)
        fw_hat = fw_hat.at[:, : kmax + 1].set(f_pos)
        fw_hat = fw_hat.at[:, nf + kmin :].set(f_neg)

    return fw_hat


# ============================================================================
# 2D Deconvolution (Vectorized - no for loops)
# ============================================================================


@partial(jax.jit, static_argnums=(3, 4, 5))
def deconvolve_shuffle_2d(
    fw_hat: jax.Array,
    phihat1: jax.Array,
    phihat2: jax.Array,
    n_modes1: int,
    n_modes2: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Deconvolve and shuffle 2D FFT output to Fourier coefficients (Type 1).

    Fully vectorized implementation using advanced indexing.

    Args:
        fw_hat: FFT of fine grid, shape (nf2, nf1) or (n_trans, nf2, nf1)
        phihat1, phihat2: Kernel Fourier coeffs for each dimension
        n_modes1, n_modes2: Number of output modes
        modeord: Mode ordering

    Returns:
        f: Fourier coefficients, shape (n_modes2, n_modes1) or (n_trans, n_modes2, n_modes1)
    """
    batched = fw_hat.ndim == 3
    if not batched:
        fw_hat = fw_hat[None, :, :]

    nf2, nf1 = fw_hat.shape[-2], fw_hat.shape[-1]

    # Build extraction indices for each dimension
    indices1 = _build_extraction_indices_1d(nf1, n_modes1, modeord)
    indices2 = _build_extraction_indices_1d(nf2, n_modes2, modeord)

    # Build deconvolution factors for each dimension
    factors1 = _build_deconv_factors_1d(phihat1, n_modes1, modeord)
    factors2 = _build_deconv_factors_1d(phihat2, n_modes2, modeord)

    # 2D deconvolution factors via outer product
    factors_2d = factors2[:, None] * factors1[None, :]  # (n_modes2, n_modes1)

    # Extract using advanced indexing: fw_hat[..., indices2, :][:, :, indices1]
    # Use meshgrid for 2D indexing
    idx2, idx1 = jnp.meshgrid(indices2, indices1, indexing="ij")

    # Extract all modes at once
    f = fw_hat[:, idx2, idx1] * factors_2d[None, :, :]

    if not batched:
        f = f[0]

    return f


@partial(jax.jit, static_argnums=(3, 4, 5))
def deconvolve_pad_2d(
    f: jax.Array,
    phihat1: jax.Array,
    phihat2: jax.Array,
    nf1: int,
    nf2: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Deconvolve and pad 2D Fourier coefficients for iFFT (Type 2).

    Fully vectorized implementation.

    Args:
        f: Fourier coefficients, shape (n_modes2, n_modes1) or (n_trans, n_modes2, n_modes1)
        phihat1, phihat2: Kernel Fourier coeffs for each dimension
        nf1, nf2: Fine grid sizes
        modeord: Mode ordering

    Returns:
        fw_hat: Padded grid for iFFT, shape (nf2, nf1) or (n_trans, nf2, nf1)
    """
    batched = f.ndim == 3
    if not batched:
        f = f[None, :, :]

    n_trans, n_modes2, n_modes1 = f.shape

    k1_min = -n_modes1 // 2
    k1_max = (n_modes1 - 1) // 2
    k2_min = -n_modes2 // 2
    k2_max = (n_modes2 - 1) // 2

    # Build deconvolution factors
    factors1 = _build_deconv_factors_1d(phihat1, n_modes1, modeord)
    factors2 = _build_deconv_factors_1d(phihat2, n_modes2, modeord)
    factors_2d = factors2[:, None] * factors1[None, :]

    # Deconvolve
    f_deconv = f * factors_2d[None, :, :]

    # Split into quadrants based on positive/negative frequencies
    if modeord == 0:
        # CMCL: f[..., i, j] -> (k2_min + i, k1_min + j)
        n_neg1 = -k1_min
        n_neg2 = -k2_min
        # Quadrants: (neg2, neg1), (neg2, pos1), (pos2, neg1), (pos2, pos1)
        f_nn = f_deconv[:, :n_neg2, :n_neg1]  # neg y, neg x
        f_np = f_deconv[:, :n_neg2, n_neg1:]  # neg y, pos x
        f_pn = f_deconv[:, n_neg2:, :n_neg1]  # pos y, neg x
        f_pp = f_deconv[:, n_neg2:, n_neg1:]  # pos y, pos x
    else:
        # FFT-style: f[..., i, j] -> output order is [0..kmax, kmin..-1]
        n_pos1 = k1_max + 1
        n_pos2 = k2_max + 1
        f_pp = f_deconv[:, :n_pos2, :n_pos1]  # pos y, pos x
        f_pn = f_deconv[:, :n_pos2, n_pos1:]  # pos y, neg x
        f_np = f_deconv[:, n_pos2:, :n_pos1]  # neg y, pos x
        f_nn = f_deconv[:, n_pos2:, n_pos1:]  # neg y, neg x

    # Initialize output
    fw_hat = jnp.zeros((n_trans, nf2, nf1), dtype=f.dtype)

    # Place quadrants in FFT order
    # Positive y (0 to k2_max), positive x (0 to k1_max)
    fw_hat = fw_hat.at[:, : k2_max + 1, : k1_max + 1].set(f_pp)
    # Positive y, negative x (nf1 + k1_min to nf1 - 1)
    fw_hat = fw_hat.at[:, : k2_max + 1, nf1 + k1_min :].set(f_pn)
    # Negative y (nf2 + k2_min to nf2 - 1), positive x
    fw_hat = fw_hat.at[:, nf2 + k2_min :, : k1_max + 1].set(f_np)
    # Negative y, negative x
    fw_hat = fw_hat.at[:, nf2 + k2_min :, nf1 + k1_min :].set(f_nn)

    if not batched:
        fw_hat = fw_hat[0]

    return fw_hat


# ============================================================================
# 3D Deconvolution (Vectorized - no for loops)
# ============================================================================


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def deconvolve_shuffle_3d(
    fw_hat: jax.Array,
    phihat1: jax.Array,
    phihat2: jax.Array,
    phihat3: jax.Array,
    n_modes1: int,
    n_modes2: int,
    n_modes3: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Deconvolve and shuffle 3D FFT output to Fourier coefficients (Type 1).

    Fully vectorized implementation using advanced indexing.

    Args:
        fw_hat: FFT of fine grid, shape (nf3, nf2, nf1) or (n_trans, nf3, nf2, nf1)
        phihat1, phihat2, phihat3: Kernel Fourier coeffs for each dimension
        n_modes1, n_modes2, n_modes3: Number of output modes
        modeord: Mode ordering

    Returns:
        f: Fourier coefficients, shape (n_modes3, n_modes2, n_modes1) or
           (n_trans, n_modes3, n_modes2, n_modes1)
    """
    batched = fw_hat.ndim == 4
    if not batched:
        fw_hat = fw_hat[None, :, :, :]

    nf3, nf2, nf1 = fw_hat.shape[-3], fw_hat.shape[-2], fw_hat.shape[-1]

    # Build extraction indices for each dimension
    indices1 = _build_extraction_indices_1d(nf1, n_modes1, modeord)
    indices2 = _build_extraction_indices_1d(nf2, n_modes2, modeord)
    indices3 = _build_extraction_indices_1d(nf3, n_modes3, modeord)

    # Build deconvolution factors for each dimension
    factors1 = _build_deconv_factors_1d(phihat1, n_modes1, modeord)
    factors2 = _build_deconv_factors_1d(phihat2, n_modes2, modeord)
    factors3 = _build_deconv_factors_1d(phihat3, n_modes3, modeord)

    # 3D deconvolution factors via outer product
    factors_3d = factors3[:, None, None] * factors2[None, :, None] * factors1[None, None, :]

    # Create 3D index meshgrid
    idx3, idx2, idx1 = jnp.meshgrid(indices3, indices2, indices1, indexing="ij")

    # Extract all modes at once using advanced indexing
    f = fw_hat[:, idx3, idx2, idx1] * factors_3d[None, :, :, :]

    if not batched:
        f = f[0]

    return f


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def deconvolve_pad_3d(
    f: jax.Array,
    phihat1: jax.Array,
    phihat2: jax.Array,
    phihat3: jax.Array,
    nf1: int,
    nf2: int,
    nf3: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Deconvolve and pad 3D Fourier coefficients for iFFT (Type 2).

    Fully vectorized implementation.

    Args:
        f: Fourier coefficients, shape (n_modes3, n_modes2, n_modes1) or
           (n_trans, n_modes3, n_modes2, n_modes1)
        phihat1, phihat2, phihat3: Kernel Fourier coeffs for each dimension
        nf1, nf2, nf3: Fine grid sizes
        modeord: Mode ordering

    Returns:
        fw_hat: Padded grid for iFFT
    """
    batched = f.ndim == 4
    if not batched:
        f = f[None, :, :, :]

    n_trans, n_modes3, n_modes2, n_modes1 = f.shape

    k1_min = -n_modes1 // 2
    k1_max = (n_modes1 - 1) // 2
    k2_min = -n_modes2 // 2
    k2_max = (n_modes2 - 1) // 2
    k3_min = -n_modes3 // 2
    k3_max = (n_modes3 - 1) // 2

    # Build deconvolution factors
    factors1 = _build_deconv_factors_1d(phihat1, n_modes1, modeord)
    factors2 = _build_deconv_factors_1d(phihat2, n_modes2, modeord)
    factors3 = _build_deconv_factors_1d(phihat3, n_modes3, modeord)
    factors_3d = factors3[:, None, None] * factors2[None, :, None] * factors1[None, None, :]

    # Deconvolve
    f_deconv = f * factors_3d[None, :, :, :]

    # Split into octants based on positive/negative frequencies
    if modeord == 0:
        n_neg1 = -k1_min
        n_neg2 = -k2_min
        n_neg3 = -k3_min
        # Extract 8 octants (nnn, nnp, npn, npp, pnn, pnp, ppn, ppp)
        f_nnn = f_deconv[:, :n_neg3, :n_neg2, :n_neg1]
        f_nnp = f_deconv[:, :n_neg3, :n_neg2, n_neg1:]
        f_npn = f_deconv[:, :n_neg3, n_neg2:, :n_neg1]
        f_npp = f_deconv[:, :n_neg3, n_neg2:, n_neg1:]
        f_pnn = f_deconv[:, n_neg3:, :n_neg2, :n_neg1]
        f_pnp = f_deconv[:, n_neg3:, :n_neg2, n_neg1:]
        f_ppn = f_deconv[:, n_neg3:, n_neg2:, :n_neg1]
        f_ppp = f_deconv[:, n_neg3:, n_neg2:, n_neg1:]
    else:
        n_pos1 = k1_max + 1
        n_pos2 = k2_max + 1
        n_pos3 = k3_max + 1
        f_ppp = f_deconv[:, :n_pos3, :n_pos2, :n_pos1]
        f_ppn = f_deconv[:, :n_pos3, :n_pos2, n_pos1:]
        f_pnp = f_deconv[:, :n_pos3, n_pos2:, :n_pos1]
        f_pnn = f_deconv[:, :n_pos3, n_pos2:, n_pos1:]
        f_npp = f_deconv[:, n_pos3:, :n_pos2, :n_pos1]
        f_npn = f_deconv[:, n_pos3:, :n_pos2, n_pos1:]
        f_nnp = f_deconv[:, n_pos3:, n_pos2:, :n_pos1]
        f_nnn = f_deconv[:, n_pos3:, n_pos2:, n_pos1:]

    # Initialize output
    fw_hat = jnp.zeros((n_trans, nf3, nf2, nf1), dtype=f.dtype)

    # Place octants in FFT order
    # ppp: positive z, positive y, positive x
    fw_hat = fw_hat.at[:, : k3_max + 1, : k2_max + 1, : k1_max + 1].set(f_ppp)
    # ppn: positive z, positive y, negative x
    fw_hat = fw_hat.at[:, : k3_max + 1, : k2_max + 1, nf1 + k1_min :].set(f_ppn)
    # pnp: positive z, negative y, positive x
    fw_hat = fw_hat.at[:, : k3_max + 1, nf2 + k2_min :, : k1_max + 1].set(f_pnp)
    # pnn: positive z, negative y, negative x
    fw_hat = fw_hat.at[:, : k3_max + 1, nf2 + k2_min :, nf1 + k1_min :].set(f_pnn)
    # npp: negative z, positive y, positive x
    fw_hat = fw_hat.at[:, nf3 + k3_min :, : k2_max + 1, : k1_max + 1].set(f_npp)
    # npn: negative z, positive y, negative x
    fw_hat = fw_hat.at[:, nf3 + k3_min :, : k2_max + 1, nf1 + k1_min :].set(f_npn)
    # nnp: negative z, negative y, positive x
    fw_hat = fw_hat.at[:, nf3 + k3_min :, nf2 + k2_min :, : k1_max + 1].set(f_nnp)
    # nnn: negative z, negative y, negative x
    fw_hat = fw_hat.at[:, nf3 + k3_min :, nf2 + k2_min :, nf1 + k1_min :].set(f_nnn)

    if not batched:
        fw_hat = fw_hat[0]

    return fw_hat
