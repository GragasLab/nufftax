"""
Type 3 NUFFT transforms: Nonuniform to Nonuniform.

Type 3 computes the Fourier transform from nonuniform source points
to nonuniform target frequencies:
    f[k] = sum_j c[j] * exp(isign * i * s[k] * x[j])

where both x[j] (source points) and s[k] (target frequencies) are nonuniform.

Algorithm (from FINUFFT):
1. Pre-phase: c' = c * exp(isign * i * D * x)
2. Rescale source points: x' = (x - C) / gamma
3. Spread c' at x' to get fw on fine grid (spatial representation)
4. Use inner Type 2: treat fw as "Fourier modes" and evaluate at s'
   (Type 2 does: deconvolve -> IFFT -> interpolate)
5. Deconvolve: f = result * (1/phihat(s')) * exp(isign * i * (s-D) * C)

JIT Compilation:
Type 3 requires the `n_modes` parameter for JIT compilation (grid sizes must
be static). Use `compute_type3_grid_size()` to pre-compute appropriate values.

Reference: FINUFFT finufft_core.cpp
"""

from functools import partial

import jax
import jax.numpy as jnp

from ..core.kernel import compute_kernel_params, es_kernel
from ..core.spread import spread_1d, spread_2d, spread_3d
from .nufft2 import nufft1d2, nufft2d2, nufft3d2


def _get_imag_unit(c: jax.Array) -> jax.Array:
    """Get imaginary unit with same dtype as c."""
    return jnp.array(1j, dtype=c.dtype)


def _next_smooth_even(n: int) -> int:
    """Find next even integer >= n that has only factors of 2, 3, 5.

    Type 3 NUFFT requires even grid sizes for proper FFT alignment.
    """
    if n <= 1:
        return 2
    if n % 2 == 1:
        n += 1
    while True:
        m = n
        while m % 2 == 0:
            m //= 2
        while m % 3 == 0:
            m //= 3
        while m % 5 == 0:
            m //= 5
        if m == 1:
            return n
        n += 2


def compute_type3_grid_size(
    x_extent: float,
    s_extent: float,
    eps: float = 1e-6,
    upsampfac: float = 2.0,
) -> int:
    """Compute appropriate grid size for 1D Type 3 NUFFT.

    This helper function can be used to pre-compute grid sizes for JIT compilation.

    Args:
        x_extent: Half-width of source points range, i.e., max(|x - center|)
        s_extent: Half-width of target frequencies range, i.e., max(|s - center|)
        eps: Requested precision
        upsampfac: Oversampling factor

    Returns:
        nf: Grid size (smooth integer with factors 2, 3, 5)

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([...])  # source points
        >>> s = jnp.array([...])  # target frequencies
        >>> x_ext = (x.max() - x.min()) / 2
        >>> s_ext = (s.max() - s.min()) / 2
        >>> nf = compute_type3_grid_size(float(x_ext), float(s_ext), eps=1e-6)
        >>> # Now use nf in JIT-compiled code:
        >>> jit_fn = jax.jit(lambda x, c, s: nufft1d3(x, c, s, n_modes=nf), static_argnums=())
    """
    import numpy as np

    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    # Ensure non-zero extents
    x_extent = max(x_extent, 1e-10)
    s_extent = max(s_extent, 1e-10)

    nf_float = 2.0 * upsampfac * s_extent * x_extent / np.pi + nspread
    return _next_smooth_even(int(np.ceil(nf_float)))


def compute_type3_grid_sizes_2d(
    x_extent: float,
    y_extent: float,
    s_extent: float,
    t_extent: float,
    eps: float = 1e-6,
    upsampfac: float = 2.0,
) -> tuple[int, int]:
    """Compute appropriate grid sizes for 2D Type 3 NUFFT.

    Args:
        x_extent, y_extent: Half-widths of source point ranges
        s_extent, t_extent: Half-widths of target frequency ranges
        eps: Requested precision
        upsampfac: Oversampling factor

    Returns:
        (nf1, nf2): Grid sizes for each dimension
    """
    import numpy as np

    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    x_extent = max(x_extent, 1e-10)
    y_extent = max(y_extent, 1e-10)
    s_extent = max(s_extent, 1e-10)
    t_extent = max(t_extent, 1e-10)

    nf1 = _next_smooth_even(int(np.ceil(2.0 * upsampfac * s_extent * x_extent / np.pi + nspread)))
    nf2 = _next_smooth_even(int(np.ceil(2.0 * upsampfac * t_extent * y_extent / np.pi + nspread)))
    return nf1, nf2


def compute_type3_grid_sizes_3d(
    x_extent: float,
    y_extent: float,
    z_extent: float,
    s_extent: float,
    t_extent: float,
    u_extent: float,
    eps: float = 1e-6,
    upsampfac: float = 2.0,
) -> tuple[int, int, int]:
    """Compute appropriate grid sizes for 3D Type 3 NUFFT.

    Args:
        x_extent, y_extent, z_extent: Half-widths of source point ranges
        s_extent, t_extent, u_extent: Half-widths of target frequency ranges
        eps: Requested precision
        upsampfac: Oversampling factor

    Returns:
        (nf1, nf2, nf3): Grid sizes for each dimension
    """
    import numpy as np

    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread

    x_extent = max(x_extent, 1e-10)
    y_extent = max(y_extent, 1e-10)
    z_extent = max(z_extent, 1e-10)
    s_extent = max(s_extent, 1e-10)
    t_extent = max(t_extent, 1e-10)
    u_extent = max(u_extent, 1e-10)

    nf1 = _next_smooth_even(int(np.ceil(2.0 * upsampfac * s_extent * x_extent / np.pi + nspread)))
    nf2 = _next_smooth_even(int(np.ceil(2.0 * upsampfac * t_extent * y_extent / np.pi + nspread)))
    nf3 = _next_smooth_even(int(np.ceil(2.0 * upsampfac * u_extent * z_extent / np.pi + nspread)))
    return nf1, nf2, nf3


def _get_real_dtype(c: jax.Array):
    """Get the real dtype corresponding to a complex dtype."""
    if c.dtype == jnp.complex64:
        return jnp.float32
    elif c.dtype == jnp.complex128:
        return jnp.float64
    else:
        # For real inputs, use the same dtype
        return c.dtype


def _kernel_ft_at_point(k: jax.Array, nspread: int, beta: float, c: float, dtype=None) -> jax.Array:
    """Evaluate Fourier transform of ES kernel at arbitrary frequency k.

    Uses quadrature to compute:
        phi_hat(k) = integral_{-J/2}^{J/2} phi(z) * cos(k*z) dz

    Args:
        k: Frequency points (can be array)
        nspread: Kernel width J
        beta: Kernel shape parameter
        c: Kernel normalization (1/J^2)
        dtype: Output dtype (default: inferred from k)

    Returns:
        phi_hat(k): Kernel Fourier transform at frequencies k
    """
    if dtype is None:
        dtype = k.dtype

    J2 = nspread / 2.0
    n_quad = max(int(4 + 3.0 * J2), 20)

    z = jnp.linspace(-J2, J2, n_quad, dtype=dtype)
    dz = z[1] - z[0]

    phi_vals = es_kernel(z, beta, c)

    # Trapezoidal weights
    w = jnp.ones(n_quad, dtype=dtype) * dz
    w = w.at[0].set(dz / 2)
    w = w.at[-1].set(dz / 2)

    k = jnp.atleast_1d(k).astype(dtype)
    phase = jnp.outer(k, z)

    phi_hat = jnp.sum(phi_vals[None, :] * w[None, :] * jnp.cos(phase), axis=1)

    return phi_hat


@partial(jax.jit, static_argnames=("eps", "isign", "upsampfac", "n_modes"))
def nufft1d3(
    x: jax.Array,
    c: jax.Array,
    s: jax.Array,
    n_modes: int,
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> jax.Array:
    """
    1D Type-3 NUFFT: nonuniform to nonuniform.

    Computes:
        f[k] = sum_{j=0}^{M-1} c[j] * exp(isign * i * s[k] * x[j])

    for k = 0, ..., N-1 where s[k] are the target frequencies.

    Use `compute_type3_grid_size()` to compute `n_modes` from data bounds.

    Args:
        x: Source points (nonuniform), shape (M,)
        c: Complex strengths at source points, shape (M,) or (n_trans, M)
        s: Target frequencies (nonuniform), shape (N,)
        n_modes: Grid size. Use compute_type3_grid_size() to compute.
        eps: Requested precision (tolerance)
        isign: Sign in the exponential (+1 or -1)
        upsampfac: Oversampling factor (default 2.0)

    Returns:
        f: Values at target frequencies, shape (N,) or (n_trans, N)
    """
    # Handle batched input
    batched = c.ndim == 2
    if not batched:
        c = c[None, :]

    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    kc = kernel_params.c

    # Compute interval centers and half-widths (traceable operations)
    x_min, x_max = jnp.min(x), jnp.max(x)
    s_min, s_max = jnp.min(s), jnp.max(s)

    C = (x_min + x_max) / 2.0
    D = (s_min + s_max) / 2.0
    S = jnp.maximum((s_max - s_min) / 2.0, 1e-10)

    # Fine grid parameters (n_modes is required for JIT)
    nf = n_modes

    h = 2.0 * jnp.pi / nf
    gamma = nf / (2.0 * upsampfac * S)

    # Rescale source points
    x_rescaled = (x - C) / gamma
    x_normalized = jnp.mod(x_rescaled + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Get dtype-appropriate imaginary unit
    imag_unit = _get_imag_unit(c)

    # Pre-phase: c' = c * exp(isign * i * D * x)
    prephase = jnp.exp(imag_unit * isign * D * x)
    c_phased = c * prephase[None, :]

    # Spread to fine grid (spatial representation)
    fw = spread_1d(x_normalized, c_phased, nf, kernel_params)

    # Rescale target frequencies: s' = h * gamma * (s - D)
    s_rescaled = h * gamma * (s - D)
    s_normalized = jnp.mod(s_rescaled + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Inner Type 2: treats fw as "Fourier modes" and evaluates at s'
    # Type 2 computes: result[j] = sum_k fw[k] * exp(isign * k * s'[j])
    # It does: deconvolve fw -> IFFT -> interpolate at s'
    f = nufft1d2(s_normalized, fw, eps=eps, isign=isign, upsampfac=upsampfac, modeord=0)

    # Deconvolution: multiply by 1/phihat(s') * phase_correction
    real_dtype = _get_real_dtype(c)
    phi_hat = _kernel_ft_at_point(s_rescaled, nspread, beta, kc, dtype=real_dtype)
    phase_correction = jnp.exp(imag_unit * isign * (s - D) * C)
    deconv = phase_correction / phi_hat
    f = f * deconv[None, :]

    if not batched:
        f = f[0]

    return f


@partial(jax.jit, static_argnames=("eps", "isign", "upsampfac", "n_modes"))
def nufft2d3(
    x: jax.Array,
    y: jax.Array,
    c: jax.Array,
    s: jax.Array,
    t: jax.Array,
    n_modes: tuple[int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> jax.Array:
    """
    2D Type-3 NUFFT: nonuniform to nonuniform.

    Computes:
        f[k] = sum_j c[j] * exp(isign * i * (s[k]*x[j] + t[k]*y[j]))

    Use `compute_type3_grid_sizes_2d()` to compute `n_modes` from data bounds.

    Args:
        x, y: Source points (nonuniform), shape (M,)
        c: Complex strengths, shape (M,) or (n_trans, M)
        s, t: Target frequencies (nonuniform), shape (N,)
        n_modes: Grid sizes (nf1, nf2). Use compute_type3_grid_sizes_2d().
        eps: Requested precision
        isign: Sign in the exponential
        upsampfac: Oversampling factor

    Returns:
        f: Values at target frequencies, shape (N,) or (n_trans, N)
    """
    batched = c.ndim == 2
    if not batched:
        c = c[None, :]

    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    kc = kernel_params.c

    # Compute intervals (traceable operations)
    x_min, x_max = jnp.min(x), jnp.max(x)
    y_min, y_max = jnp.min(y), jnp.max(y)
    s_min, s_max = jnp.min(s), jnp.max(s)
    t_min, t_max = jnp.min(t), jnp.max(t)

    Cx = (x_min + x_max) / 2.0
    Cy = (y_min + y_max) / 2.0
    Ds = (s_min + s_max) / 2.0
    Ss = jnp.maximum((s_max - s_min) / 2.0, 1e-10)
    Dt = (t_min + t_max) / 2.0
    St = jnp.maximum((t_max - t_min) / 2.0, 1e-10)

    # Fine grid parameters (n_modes is required for JIT)
    nf1, nf2 = n_modes

    h1, h2 = 2.0 * jnp.pi / nf1, 2.0 * jnp.pi / nf2
    gamma1, gamma2 = nf1 / (2.0 * upsampfac * Ss), nf2 / (2.0 * upsampfac * St)

    # Rescale source points
    x_normalized = jnp.mod((x - Cx) / gamma1 + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    y_normalized = jnp.mod((y - Cy) / gamma2 + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Get dtype-appropriate imaginary unit
    imag_unit = _get_imag_unit(c)

    # Pre-phase
    prephase = jnp.exp(imag_unit * isign * (Ds * x + Dt * y))
    c_phased = c * prephase[None, :]

    # Spread
    fw = spread_2d(x_normalized, y_normalized, c_phased, nf1, nf2, kernel_params)

    # Rescale target frequencies
    s_rescaled = h1 * gamma1 * (s - Ds)
    t_rescaled = h2 * gamma2 * (t - Dt)
    s_normalized = jnp.mod(s_rescaled + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    t_normalized = jnp.mod(t_rescaled + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Inner Type 2
    f = nufft2d2(s_normalized, t_normalized, fw, eps=eps, isign=isign, upsampfac=upsampfac)

    # Deconvolution
    real_dtype = _get_real_dtype(c)
    phi_hat1 = _kernel_ft_at_point(s_rescaled, nspread, beta, kc, dtype=real_dtype)
    phi_hat2 = _kernel_ft_at_point(t_rescaled, nspread, beta, kc, dtype=real_dtype)
    phase_correction = jnp.exp(imag_unit * isign * ((s - Ds) * Cx + (t - Dt) * Cy))
    f = f * phase_correction[None, :] / (phi_hat1 * phi_hat2)[None, :]

    if not batched:
        f = f[0]

    return f


@partial(jax.jit, static_argnames=("eps", "isign", "upsampfac", "n_modes"))
def nufft3d3(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    c: jax.Array,
    s: jax.Array,
    t: jax.Array,
    u: jax.Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> jax.Array:
    """
    3D Type-3 NUFFT: nonuniform to nonuniform.

    Computes:
        f[k] = sum_j c[j] * exp(isign * i * (s[k]*x[j] + t[k]*y[j] + u[k]*z[j]))

    Use `compute_type3_grid_sizes_3d()` to compute `n_modes` from data bounds.

    Args:
        x, y, z: Source points (nonuniform), shape (M,)
        c: Complex strengths, shape (M,) or (n_trans, M)
        s, t, u: Target frequencies (nonuniform), shape (N,)
        n_modes: Grid sizes (nf1, nf2, nf3). Use compute_type3_grid_sizes_3d().
        eps: Requested precision
        isign: Sign in the exponential
        upsampfac: Oversampling factor

    Returns:
        f: Values at target frequencies, shape (N,) or (n_trans, N)
    """
    batched = c.ndim == 2
    if not batched:
        c = c[None, :]

    kernel_params = compute_kernel_params(eps, upsampfac)
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    kc = kernel_params.c

    # Compute intervals (traceable operations)
    x_min, x_max = jnp.min(x), jnp.max(x)
    y_min, y_max = jnp.min(y), jnp.max(y)
    z_min, z_max = jnp.min(z), jnp.max(z)
    s_min, s_max = jnp.min(s), jnp.max(s)
    t_min, t_max = jnp.min(t), jnp.max(t)
    u_min, u_max = jnp.min(u), jnp.max(u)

    Cx = (x_min + x_max) / 2.0
    Cy = (y_min + y_max) / 2.0
    Cz = (z_min + z_max) / 2.0
    Ds = (s_min + s_max) / 2.0
    Ss = jnp.maximum((s_max - s_min) / 2.0, 1e-10)
    Dt = (t_min + t_max) / 2.0
    St = jnp.maximum((t_max - t_min) / 2.0, 1e-10)
    Du = (u_min + u_max) / 2.0
    Su = jnp.maximum((u_max - u_min) / 2.0, 1e-10)

    # Fine grid parameters (n_modes is required for JIT)
    nf1, nf2, nf3 = n_modes

    h1, h2, h3 = 2.0 * jnp.pi / nf1, 2.0 * jnp.pi / nf2, 2.0 * jnp.pi / nf3
    gamma1 = nf1 / (2.0 * upsampfac * Ss)
    gamma2 = nf2 / (2.0 * upsampfac * St)
    gamma3 = nf3 / (2.0 * upsampfac * Su)

    # Rescale source points
    x_normalized = jnp.mod((x - Cx) / gamma1 + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    y_normalized = jnp.mod((y - Cy) / gamma2 + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    z_normalized = jnp.mod((z - Cz) / gamma3 + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Get dtype-appropriate imaginary unit
    imag_unit = _get_imag_unit(c)

    # Pre-phase
    prephase = jnp.exp(imag_unit * isign * (Ds * x + Dt * y + Du * z))
    c_phased = c * prephase[None, :]

    # Spread
    fw = spread_3d(x_normalized, y_normalized, z_normalized, c_phased, nf1, nf2, nf3, kernel_params)

    # Rescale target frequencies
    s_rescaled = h1 * gamma1 * (s - Ds)
    t_rescaled = h2 * gamma2 * (t - Dt)
    u_rescaled = h3 * gamma3 * (u - Du)
    s_normalized = jnp.mod(s_rescaled + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    t_normalized = jnp.mod(t_rescaled + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    u_normalized = jnp.mod(u_rescaled + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Inner Type 2
    f = nufft3d2(
        s_normalized,
        t_normalized,
        u_normalized,
        fw,
        eps=eps,
        isign=isign,
        upsampfac=upsampfac,
    )

    # Deconvolution
    real_dtype = _get_real_dtype(c)
    phi_hat1 = _kernel_ft_at_point(s_rescaled, nspread, beta, kc, dtype=real_dtype)
    phi_hat2 = _kernel_ft_at_point(t_rescaled, nspread, beta, kc, dtype=real_dtype)
    phi_hat3 = _kernel_ft_at_point(u_rescaled, nspread, beta, kc, dtype=real_dtype)
    phase_correction = jnp.exp(imag_unit * isign * ((s - Ds) * Cx + (t - Dt) * Cy + (u - Du) * Cz))
    f = f * phase_correction[None, :] / (phi_hat1 * phi_hat2 * phi_hat3)[None, :]

    if not batched:
        f = f[0]

    return f
