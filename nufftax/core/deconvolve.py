"""
Deconvolution functions for NUFFT.

The deconvolution step corrects for the spreading kernel by dividing
the Fourier coefficients by the kernel's Fourier transform.

For Type 1: deconvolve and shuffle from fine grid FFT to output modes
For Type 2: deconvolve and pad from input modes to fine grid for iFFT

Reference: FINUFFT src/finufft_core.cpp (deconvolveshuffle1d, etc.)
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(2, 3))
def deconvolve_shuffle_1d(
    fw_hat: jax.Array,
    phihat: jax.Array,
    n_modes: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Deconvolve and shuffle 1D FFT output to Fourier coefficients (Type 1).

    Applies deconvolution (division by kernel Fourier coeffs) and shuffles
    from FFT ordering to output mode ordering.

    Args:
        fw_hat: FFT of fine grid, shape (nf,) or (n_trans, nf)
        phihat: Kernel Fourier coefficients, shape (nf//2 + 1,)
        n_modes: Number of output modes
        modeord: Mode ordering (0=CMCL, 1=FFT-style)

    Returns:
        f: Fourier coefficients, shape (n_modes,) or (n_trans, n_modes)
    """
    batched = fw_hat.ndim == 2
    if not batched:
        fw_hat = fw_hat[None, :]

    nf = fw_hat.shape[-1]

    # Frequency ranges for output modes
    # CMCL ordering: from -n_modes//2 to (n_modes-1)//2
    kmin = -n_modes // 2
    kmax = (n_modes - 1) // 2

    # Positive frequencies: k = 0, 1, ..., kmax
    # These are at indices 0, 1, ..., kmax in fw_hat
    f_pos = fw_hat[..., : kmax + 1] / phihat[: kmax + 1]

    # Negative frequencies: k = kmin, ..., -1
    # These are at indices nf+kmin, ..., nf-1 in fw_hat
    # The kernel is symmetric, so phihat[-k] = phihat[k]
    neg_indices = jnp.arange(nf + kmin, nf)
    f_neg = fw_hat[..., neg_indices] / phihat[1 : -kmin + 1][::-1]

    if modeord == 0:
        # CMCL ordering: negative frequencies first, then positive
        f = jnp.concatenate([f_neg, f_pos], axis=-1)
    else:
        # FFT-style ordering: positive frequencies first, then negative
        f = jnp.concatenate([f_pos, f_neg], axis=-1)

    if not batched:
        f = f[0]

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

    Applies deconvolution (division by kernel Fourier coeffs) and pads
    with zeros for the fine grid iFFT.

    Args:
        f: Fourier coefficients, shape (n_modes,) or (n_trans, n_modes)
        phihat: Kernel Fourier coefficients, shape (nf//2 + 1,)
        nf: Fine grid size
        modeord: Mode ordering (0=CMCL, 1=FFT-style)

    Returns:
        fw_hat: Padded and deconvolved grid for iFFT, shape (nf,) or (n_trans, nf)
    """
    batched = f.ndim == 2
    if not batched:
        f = f[None, :]

    n_trans, n_modes = f.shape

    # Frequency ranges
    kmin = -n_modes // 2
    kmax = (n_modes - 1) // 2

    if modeord == 0:
        # CMCL ordering: f contains [f[kmin], ..., f[-1], f[0], ..., f[kmax]]
        n_neg = -kmin  # Number of negative frequencies
        f_neg = f[..., :n_neg]
        f_pos = f[..., n_neg:]
    else:
        # FFT-style ordering: f contains [f[0], ..., f[kmax], f[kmin], ..., f[-1]]
        n_pos = kmax + 1
        f_pos = f[..., :n_pos]
        f_neg = f[..., n_pos:]

    # Deconvolve positive frequencies
    f_pos_deconv = f_pos / phihat[: kmax + 1]

    # Deconvolve negative frequencies (kernel is symmetric)
    f_neg_deconv = f_neg / phihat[1 : -kmin + 1][::-1]

    # Initialize output with zeros
    fw_hat = jnp.zeros((n_trans, nf), dtype=f.dtype)

    # Place positive frequencies at indices 0, 1, ..., kmax
    fw_hat = fw_hat.at[..., : kmax + 1].set(f_pos_deconv)

    # Place negative frequencies at indices nf+kmin, ..., nf-1
    fw_hat = fw_hat.at[..., nf + kmin :].set(f_neg_deconv)

    if not batched:
        fw_hat = fw_hat[0]

    return fw_hat


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

    n_trans, nf2, nf1 = fw_hat.shape

    # Process each y-row using 1D deconvolution
    # First, extract and deconvolve along x (dim 1)
    k1_min = -n_modes1 // 2
    k1_max = (n_modes1 - 1) // 2
    k2_min = -n_modes2 // 2
    k2_max = (n_modes2 - 1) // 2

    # Build the output array
    f = jnp.zeros((n_trans, n_modes2, n_modes1), dtype=fw_hat.dtype)

    # Deconvolution factors for x dimension
    deconv_pos_x = 1.0 / phihat1[: k1_max + 1]
    deconv_neg_x = 1.0 / phihat1[1 : -k1_min + 1][::-1]

    # Deconvolution factors for y dimension
    deconv_pos_y = 1.0 / phihat2[: k2_max + 1]
    deconv_neg_y = 1.0 / phihat2[1 : -k2_min + 1][::-1]

    # Process positive y frequencies (k2 = 0, 1, ..., k2_max)
    for k2 in range(k2_max + 1):
        row = fw_hat[:, k2, :]  # (n_trans, nf1)
        # Extract and deconvolve x
        f_pos = row[:, : k1_max + 1] * deconv_pos_x
        f_neg = row[:, nf1 + k1_min :] * deconv_neg_x
        # Deconvolve y
        y_factor = deconv_pos_y[k2]
        if modeord == 0:
            f = f.at[:, -k2_min + k2, :].set(jnp.concatenate([f_neg * y_factor, f_pos * y_factor], axis=-1))
        else:
            f = f.at[:, k2, :].set(jnp.concatenate([f_pos * y_factor, f_neg * y_factor], axis=-1))

    # Process negative y frequencies (k2 = k2_min, ..., -1)
    for k2 in range(k2_min, 0):
        row = fw_hat[:, nf2 + k2, :]
        f_pos = row[:, : k1_max + 1] * deconv_pos_x
        f_neg = row[:, nf1 + k1_min :] * deconv_neg_x
        y_factor = deconv_neg_y[-k2 - 1]
        if modeord == 0:
            f = f.at[:, k2 - k2_min, :].set(jnp.concatenate([f_neg * y_factor, f_pos * y_factor], axis=-1))
        else:
            f = f.at[:, n_modes2 + k2, :].set(jnp.concatenate([f_pos * y_factor, f_neg * y_factor], axis=-1))

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

    # Deconvolution factors
    deconv_pos_x = 1.0 / phihat1[: k1_max + 1]
    deconv_neg_x = 1.0 / phihat1[1 : -k1_min + 1][::-1]
    deconv_pos_y = 1.0 / phihat2[: k2_max + 1]
    deconv_neg_y = 1.0 / phihat2[1 : -k2_min + 1][::-1]

    # Initialize output
    fw_hat = jnp.zeros((n_trans, nf2, nf1), dtype=f.dtype)

    if modeord == 0:
        # CMCL: f[..., i, j] corresponds to (k2_min + i, k1_min + j)
        n_neg_x = -k1_min
    else:
        n_neg_x = 0

    # Process positive y frequencies
    for k2 in range(k2_max + 1):
        if modeord == 0:
            row = f[:, -k2_min + k2, :]
            f_neg = row[:, :n_neg_x] * deconv_neg_x
            f_pos = row[:, n_neg_x:] * deconv_pos_x
        else:
            row = f[:, k2, :]
            n_pos_x = k1_max + 1
            f_pos = row[:, :n_pos_x] * deconv_pos_x
            f_neg = row[:, n_pos_x:] * deconv_neg_x

        y_factor = deconv_pos_y[k2]
        fw_hat = fw_hat.at[:, k2, : k1_max + 1].set(f_pos * y_factor)
        fw_hat = fw_hat.at[:, k2, nf1 + k1_min :].set(f_neg * y_factor)

    # Process negative y frequencies
    for k2 in range(k2_min, 0):
        if modeord == 0:
            row = f[:, k2 - k2_min, :]
            f_neg = row[:, :n_neg_x] * deconv_neg_x
            f_pos = row[:, n_neg_x:] * deconv_pos_x
        else:
            row = f[:, n_modes2 + k2, :]
            n_pos_x = k1_max + 1
            f_pos = row[:, :n_pos_x] * deconv_pos_x
            f_neg = row[:, n_pos_x:] * deconv_neg_x

        y_factor = deconv_neg_y[-k2 - 1]
        fw_hat = fw_hat.at[:, nf2 + k2, : k1_max + 1].set(f_pos * y_factor)
        fw_hat = fw_hat.at[:, nf2 + k2, nf1 + k1_min :].set(f_neg * y_factor)

    if not batched:
        fw_hat = fw_hat[0]

    return fw_hat


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

    n_trans, nf3, nf2, nf1 = fw_hat.shape

    k1_min = -n_modes1 // 2
    k1_max = (n_modes1 - 1) // 2
    k2_min = -n_modes2 // 2
    k2_max = (n_modes2 - 1) // 2
    k3_min = -n_modes3 // 2
    k3_max = (n_modes3 - 1) // 2

    # Deconvolution factors
    deconv_pos_x = 1.0 / phihat1[: k1_max + 1]
    deconv_neg_x = 1.0 / phihat1[1 : -k1_min + 1][::-1]
    deconv_pos_y = 1.0 / phihat2[: k2_max + 1]
    deconv_neg_y = 1.0 / phihat2[1 : -k2_min + 1][::-1]
    deconv_pos_z = 1.0 / phihat3[: k3_max + 1]
    deconv_neg_z = 1.0 / phihat3[1 : -k3_min + 1][::-1]

    f = jnp.zeros((n_trans, n_modes3, n_modes2, n_modes1), dtype=fw_hat.dtype)

    # Process all z slices
    for k3 in range(k3_max + 1):
        z_factor = deconv_pos_z[k3]
        z_out = -k3_min + k3 if modeord == 0 else k3

        for k2 in range(k2_max + 1):
            y_factor = deconv_pos_y[k2] * z_factor
            y_out = -k2_min + k2 if modeord == 0 else k2
            row = fw_hat[:, k3, k2, :]
            f_pos = row[:, : k1_max + 1] * deconv_pos_x * y_factor
            f_neg = row[:, nf1 + k1_min :] * deconv_neg_x * y_factor
            if modeord == 0:
                f = f.at[:, z_out, y_out, :].set(jnp.concatenate([f_neg, f_pos], axis=-1))
            else:
                f = f.at[:, z_out, y_out, :].set(jnp.concatenate([f_pos, f_neg], axis=-1))

        for k2 in range(k2_min, 0):
            y_factor = deconv_neg_y[-k2 - 1] * z_factor
            y_out = k2 - k2_min if modeord == 0 else n_modes2 + k2
            row = fw_hat[:, k3, nf2 + k2, :]
            f_pos = row[:, : k1_max + 1] * deconv_pos_x * y_factor
            f_neg = row[:, nf1 + k1_min :] * deconv_neg_x * y_factor
            if modeord == 0:
                f = f.at[:, z_out, y_out, :].set(jnp.concatenate([f_neg, f_pos], axis=-1))
            else:
                f = f.at[:, z_out, y_out, :].set(jnp.concatenate([f_pos, f_neg], axis=-1))

    for k3 in range(k3_min, 0):
        z_factor = deconv_neg_z[-k3 - 1]
        z_out = k3 - k3_min if modeord == 0 else n_modes3 + k3

        for k2 in range(k2_max + 1):
            y_factor = deconv_pos_y[k2] * z_factor
            y_out = -k2_min + k2 if modeord == 0 else k2
            row = fw_hat[:, nf3 + k3, k2, :]
            f_pos = row[:, : k1_max + 1] * deconv_pos_x * y_factor
            f_neg = row[:, nf1 + k1_min :] * deconv_neg_x * y_factor
            if modeord == 0:
                f = f.at[:, z_out, y_out, :].set(jnp.concatenate([f_neg, f_pos], axis=-1))
            else:
                f = f.at[:, z_out, y_out, :].set(jnp.concatenate([f_pos, f_neg], axis=-1))

        for k2 in range(k2_min, 0):
            y_factor = deconv_neg_y[-k2 - 1] * z_factor
            y_out = k2 - k2_min if modeord == 0 else n_modes2 + k2
            row = fw_hat[:, nf3 + k3, nf2 + k2, :]
            f_pos = row[:, : k1_max + 1] * deconv_pos_x * y_factor
            f_neg = row[:, nf1 + k1_min :] * deconv_neg_x * y_factor
            if modeord == 0:
                f = f.at[:, z_out, y_out, :].set(jnp.concatenate([f_neg, f_pos], axis=-1))
            else:
                f = f.at[:, z_out, y_out, :].set(jnp.concatenate([f_pos, f_neg], axis=-1))

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

    # Deconvolution factors
    deconv_pos_x = 1.0 / phihat1[: k1_max + 1]
    deconv_neg_x = 1.0 / phihat1[1 : -k1_min + 1][::-1]
    deconv_pos_y = 1.0 / phihat2[: k2_max + 1]
    deconv_neg_y = 1.0 / phihat2[1 : -k2_min + 1][::-1]
    deconv_pos_z = 1.0 / phihat3[: k3_max + 1]
    deconv_neg_z = 1.0 / phihat3[1 : -k3_min + 1][::-1]

    fw_hat = jnp.zeros((n_trans, nf3, nf2, nf1), dtype=f.dtype)

    n_neg_x = -k1_min if modeord == 0 else 0

    # Helper to extract f_pos, f_neg from a row
    def extract_x_components(row):
        if modeord == 0:
            f_neg = row[:, :n_neg_x] * deconv_neg_x
            f_pos = row[:, n_neg_x:] * deconv_pos_x
        else:
            n_pos_x = k1_max + 1
            f_pos = row[:, :n_pos_x] * deconv_pos_x
            f_neg = row[:, n_pos_x:] * deconv_neg_x
        return f_pos, f_neg

    for k3 in range(k3_max + 1):
        z_factor = deconv_pos_z[k3]
        z_in = -k3_min + k3 if modeord == 0 else k3

        for k2 in range(k2_max + 1):
            y_factor = deconv_pos_y[k2] * z_factor
            y_in = -k2_min + k2 if modeord == 0 else k2
            row = f[:, z_in, y_in, :]
            f_pos, f_neg = extract_x_components(row)
            fw_hat = fw_hat.at[:, k3, k2, : k1_max + 1].set(f_pos * y_factor)
            fw_hat = fw_hat.at[:, k3, k2, nf1 + k1_min :].set(f_neg * y_factor)

        for k2 in range(k2_min, 0):
            y_factor = deconv_neg_y[-k2 - 1] * z_factor
            y_in = k2 - k2_min if modeord == 0 else n_modes2 + k2
            row = f[:, z_in, y_in, :]
            f_pos, f_neg = extract_x_components(row)
            fw_hat = fw_hat.at[:, k3, nf2 + k2, : k1_max + 1].set(f_pos * y_factor)
            fw_hat = fw_hat.at[:, k3, nf2 + k2, nf1 + k1_min :].set(f_neg * y_factor)

    for k3 in range(k3_min, 0):
        z_factor = deconv_neg_z[-k3 - 1]
        z_in = k3 - k3_min if modeord == 0 else n_modes3 + k3

        for k2 in range(k2_max + 1):
            y_factor = deconv_pos_y[k2] * z_factor
            y_in = -k2_min + k2 if modeord == 0 else k2
            row = f[:, z_in, y_in, :]
            f_pos, f_neg = extract_x_components(row)
            fw_hat = fw_hat.at[:, nf3 + k3, k2, : k1_max + 1].set(f_pos * y_factor)
            fw_hat = fw_hat.at[:, nf3 + k3, k2, nf1 + k1_min :].set(f_neg * y_factor)

        for k2 in range(k2_min, 0):
            y_factor = deconv_neg_y[-k2 - 1] * z_factor
            y_in = k2 - k2_min if modeord == 0 else n_modes2 + k2
            row = f[:, z_in, y_in, :]
            f_pos, f_neg = extract_x_components(row)
            fw_hat = fw_hat.at[:, nf3 + k3, nf2 + k2, : k1_max + 1].set(f_pos * y_factor)
            fw_hat = fw_hat.at[:, nf3 + k3, nf2 + k2, nf1 + k1_min :].set(f_neg * y_factor)

    if not batched:
        fw_hat = fw_hat[0]

    return fw_hat


# ============================================================================
# Vectorized deconvolution functions (more efficient for JAX)
# ============================================================================


def _build_deconv_factors_1d(phihat: jax.Array, n_modes: int, modeord: int = 0):
    """Build deconvolution factor array for 1D case."""
    kmin = -n_modes // 2
    kmax = (n_modes - 1) // 2

    # Build factor array in output mode order
    if modeord == 0:
        # CMCL: [kmin, ..., -1, 0, 1, ..., kmax]
        # Negative frequencies first
        factors_neg = 1.0 / phihat[1 : -kmin + 1][::-1]  # for k = kmin, ..., -1
        factors_pos = 1.0 / phihat[: kmax + 1]  # for k = 0, ..., kmax
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
        # CMCL: output [kmin, ..., -1, 0, ..., kmax]
        indices = jnp.concatenate([idx_neg, idx_pos])
    else:
        # FFT-style: output [0, ..., kmax, kmin, ..., -1]
        indices = jnp.concatenate([idx_pos, idx_neg])

    return indices


@partial(jax.jit, static_argnums=(2, 3))
def deconvolve_shuffle_1d_vectorized(
    fw_hat: jax.Array,
    phihat: jax.Array,
    n_modes: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Vectorized 1D deconvolution and shuffling (Type 1).

    More efficient than the loop-based version for large arrays.
    """
    nf = fw_hat.shape[-1]

    # Build indices and factors
    indices = _build_extraction_indices_1d(nf, n_modes, modeord)
    factors = _build_deconv_factors_1d(phihat, n_modes, modeord)

    # Extract and deconvolve in one step
    f = fw_hat[..., indices] * factors

    return f


@partial(jax.jit, static_argnums=(2, 3))
def deconvolve_pad_1d_vectorized(
    f: jax.Array,
    phihat: jax.Array,
    nf: int,
    modeord: int = 0,
) -> jax.Array:
    """
    Vectorized 1D deconvolution and padding (Type 2).
    """
    n_modes = f.shape[-1]
    kmin = -n_modes // 2
    kmax = (n_modes - 1) // 2

    # Build factors in input order
    factors = _build_deconv_factors_1d(phihat, n_modes, modeord)

    # Deconvolve
    f_deconv = f * factors

    # Build placement indices
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
