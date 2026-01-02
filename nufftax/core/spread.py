"""
Spreading and interpolation operations for NUFFT.

Spreading (Type 1): Scatter nonuniform point values to a uniform grid.
Interpolation (Type 2): Gather uniform grid values at nonuniform points.

These are the computationally intensive core operations of NUFFT.
The implementation uses pure JAX operations to enable automatic differentiation.

Reference: FINUFFT src/spreadinterp.cpp
"""

from functools import partial

import jax
import jax.numpy as jnp

from .kernel import KernelParams, es_kernel, es_kernel_with_derivative

# ============================================================================
# Helper functions
# ============================================================================


def fold_rescale(x: jax.Array, n: int) -> jax.Array:
    """
    Fold and rescale coordinates from [-pi, pi) to [0, N).

    This maps nonuniform points from the standard NUFFT domain to
    grid indices suitable for spreading/interpolation.

    Args:
        x: Coordinates in [-pi, pi)
        n: Grid size

    Returns:
        Rescaled coordinates in [0, N)
    """
    # Map from [-pi, pi) to [0, 1) then to [0, N)
    inv_2pi = 1.0 / (2.0 * jnp.pi)
    result = x * inv_2pi + 0.5
    # Periodic wrapping to [0, 1)
    result = result - jnp.floor(result)
    return result * n


def _prepare_batched_c(c: jax.Array) -> tuple[jax.Array, int, bool]:
    """Prepare strengths array for batched processing.

    Args:
        c: Strengths array, shape (M,) or (n_trans, M)

    Returns:
        c_flat: Batched array, shape (n_trans, M)
        n_trans: Number of transforms
        is_batched: Whether input was already batched
    """
    if c.ndim == 2:
        return c, c.shape[0], True
    return c[None, :], 1, False


def _prepare_batched_grid_1d(fw: jax.Array) -> tuple[jax.Array, int, int, bool]:
    """Prepare 1D grid array for batched processing.

    Args:
        fw: Grid array, shape (nf,) or (n_trans, nf)

    Returns:
        fw_flat: Batched array, shape (n_trans, nf)
        nf: Grid size
        n_trans: Number of transforms
        is_batched: Whether input was already batched
    """
    if fw.ndim == 2:
        return fw, fw.shape[1], fw.shape[0], True
    return fw[None, :], fw.shape[0], 1, False


def _prepare_batched_grid_2d(fw: jax.Array) -> tuple[jax.Array, int, int, int, bool]:
    """Prepare 2D grid array for batched processing.

    Args:
        fw: Grid array, shape (nf1, nf2) or (n_trans, nf1, nf2)

    Returns:
        fw_flat: Flattened batched array, shape (n_trans, nf1*nf2)
        nf1, nf2: Grid sizes
        n_trans: Number of transforms
        is_batched: Whether input was already batched
    """
    if fw.ndim == 3:
        n_trans, nf1, nf2 = fw.shape
        return fw.reshape(n_trans, nf1 * nf2), nf1, nf2, n_trans, True
    nf1, nf2 = fw.shape
    return fw.reshape(1, nf1 * nf2), nf1, nf2, 1, False


def _prepare_batched_grid_3d(fw: jax.Array) -> tuple[jax.Array, int, int, int, int, bool]:
    """Prepare 3D grid array for batched processing.

    Args:
        fw: Grid array, shape (nf1, nf2, nf3) or (n_trans, nf1, nf2, nf3)

    Returns:
        fw_flat: Flattened batched array, shape (n_trans, nf1*nf2*nf3)
        nf1, nf2, nf3: Grid sizes
        n_trans: Number of transforms
        is_batched: Whether input was already batched
    """
    if fw.ndim == 4:
        n_trans, nf1, nf2, nf3 = fw.shape
        return fw.reshape(n_trans, nf1 * nf2 * nf3), nf1, nf2, nf3, n_trans, True
    nf1, nf2, nf3 = fw.shape
    return fw.reshape(1, nf1 * nf2 * nf3), nf1, nf2, nf3, 1, False


def compute_kernel_weights_1d(
    x_scaled: jax.Array,
    nf: int,
    kernel_params: KernelParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute kernel weights and grid indices for 1D spreading/interpolation.

    For each nonuniform point, computes:
    - The nspread grid indices it affects
    - The corresponding kernel weights

    Args:
        x_scaled: Scaled coordinates in [0, nf), shape (M,)
        nf: Fine grid size
        kernel_params: Kernel parameters

    Returns:
        indices: Grid indices, shape (M, nspread)
        weights: Kernel weights, shape (M, nspread)
    """
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    c = kernel_params.c

    # Half kernel width
    ns2 = nspread / 2.0

    # Find the leftmost grid point for each nonuniform point
    # Use ceil to match FINUFFT convention
    i0 = jnp.ceil(x_scaled - ns2).astype(jnp.int32)

    # Compute offsets from leftmost point: 0, 1, ..., nspread-1
    offsets = jnp.arange(nspread)

    # Grid indices for all kernel points: shape (M, nspread)
    indices = i0[:, None] + offsets[None, :]

    # Wrap indices periodically
    indices_wrapped = indices % nf

    # Compute z values for kernel evaluation
    # z = (grid_index - x_scaled) normalized to kernel support
    # The kernel is evaluated at z in [-ns/2, ns/2]
    z = indices.astype(x_scaled.dtype) - x_scaled[:, None]

    # Evaluate kernel
    weights = es_kernel(z, beta, c)

    return indices_wrapped, weights


def compute_kernel_weights_derivative_1d(
    x_scaled: jax.Array,
    nf: int,
    kernel_params: KernelParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute kernel weights, their derivatives, and grid indices.

    Args:
        x_scaled: Scaled coordinates in [0, nf), shape (M,)
        nf: Fine grid size
        kernel_params: Kernel parameters

    Returns:
        indices: Grid indices, shape (M, nspread)
        weights: Kernel weights, shape (M, nspread)
        dweights: Kernel weight derivatives w.r.t. x, shape (M, nspread)
    """
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    c = kernel_params.c

    ns2 = nspread / 2.0
    i0 = jnp.ceil(x_scaled - ns2).astype(jnp.int32)
    offsets = jnp.arange(nspread)
    indices = i0[:, None] + offsets[None, :]
    indices_wrapped = indices % nf

    z = indices.astype(x_scaled.dtype) - x_scaled[:, None]

    # Use fused computation for efficiency (computes both in single pass)
    weights, dweights_dz = es_kernel_with_derivative(z, beta, c)
    # Derivative of kernel w.r.t. z, but we need w.r.t. x
    # Since z = grid_idx - x_scaled, dz/dx = -1 (in grid units)
    # And x_scaled = x * nf / (2*pi), so dx_scaled/dx = nf / (2*pi)
    # The negative sign accounts for z = idx - x_scaled
    # The scaling nf/(2*pi) converts from grid to original coordinates
    scale = nf / (2.0 * jnp.pi)
    dweights = -dweights_dz * scale

    return indices_wrapped, weights, dweights


# ============================================================================
# 1D Spreading and Interpolation
# ============================================================================


def spread_1d_impl(
    x: jax.Array,
    c: jax.Array,
    nf: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    1D spreading implementation: scatter nonuniform values to grid.

    Mathematical operation:
        fw[k] = sum_j c[j] * phi(k - x[j] * nf / (2*pi))

    where phi is the spreading kernel.

    Args:
        x: Nonuniform point coordinates in [-pi, pi), shape (M,)
        c: Complex strengths at nonuniform points, shape (M,) or (n_trans, M)
        nf: Fine grid size
        kernel_params: Kernel parameters

    Returns:
        fw: Fine grid values, shape (nf,) or (n_trans, nf)
    """
    c_flat, n_trans, is_batched = _prepare_batched_c(c)

    # Scale coordinates to grid units
    x_scaled = fold_rescale(x, nf)

    # Compute kernel weights and indices
    indices, weights = compute_kernel_weights_1d(x_scaled, nf, kernel_params)
    # indices: (M, nspread), weights: (M, nspread)

    # Initialize output grid
    fw = jnp.zeros((n_trans, nf), dtype=c.dtype)

    # For each transform and each nonuniform point, accumulate contribution
    # weighted_c[t, j, k] = c[t, j] * weights[j, k]
    weighted_c = c_flat[:, :, None] * weights[None, :, :]  # (n_trans, M, nspread)

    # Flatten for segment_sum
    # indices_flat: (M * nspread,)
    indices_flat = indices.ravel()
    # weighted_c_flat: (n_trans, M * nspread)
    weighted_c_flat = weighted_c.reshape(n_trans, -1)

    # Use segment_sum for efficient accumulation (faster than add.at)
    def segment_sum_for_one_transform(wc_t):
        return jax.ops.segment_sum(wc_t, indices_flat, num_segments=nf)

    fw = jax.vmap(segment_sum_for_one_transform)(weighted_c_flat)

    if not is_batched:
        fw = fw[0]

    return fw


def interp_1d_impl(
    x: jax.Array,
    fw: jax.Array,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    1D interpolation implementation: gather grid values at nonuniform points.

    Mathematical operation:
        c[j] = sum_k fw[k] * phi(k - x[j] * nf / (2*pi))

    Args:
        x: Nonuniform point coordinates in [-pi, pi), shape (M,)
        fw: Fine grid values, shape (nf,) or (n_trans, nf)
        kernel_params: Kernel parameters

    Returns:
        c: Interpolated values at nonuniform points, shape (M,) or (n_trans, M)
    """
    fw_flat, nf, _, is_batched = _prepare_batched_grid_1d(fw)

    # Scale coordinates to grid units
    x_scaled = fold_rescale(x, nf)

    # Compute kernel weights and indices
    indices, weights = compute_kernel_weights_1d(x_scaled, nf, kernel_params)
    # indices: (M, nspread), weights: (M, nspread)

    # Gather grid values at kernel support points
    # fw_gathered[t, j, k] = fw[t, indices[j, k]]
    fw_gathered = fw_flat[:, indices]  # (n_trans, M, nspread)

    # Apply kernel weights and sum over kernel support
    # c[t, j] = sum_k fw_gathered[t, j, k] * weights[j, k]
    c = jnp.sum(fw_gathered * weights[None, :, :], axis=-1)  # (n_trans, M)

    if not is_batched:
        c = c[0]

    return c


# ============================================================================
# 2D Spreading and Interpolation
# ============================================================================


def compute_kernel_weights_2d(
    x_scaled: jax.Array,
    y_scaled: jax.Array,
    nf1: int,
    nf2: int,
    kernel_params: KernelParams,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute kernel weights and grid indices for 2D spreading/interpolation.

    Returns separate 1D indices and weights for each dimension, to be combined
    via outer product.

    Args:
        x_scaled: Scaled x coordinates in [0, nf1), shape (M,)
        y_scaled: Scaled y coordinates in [0, nf2), shape (M,)
        nf1, nf2: Fine grid sizes
        kernel_params: Kernel parameters

    Returns:
        indices_x: X grid indices, shape (M, nspread)
        indices_y: Y grid indices, shape (M, nspread)
        weights_x: X kernel weights, shape (M, nspread)
        weights_y: Y kernel weights, shape (M, nspread)
    """
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    c = kernel_params.c

    ns2 = nspread / 2.0
    offsets = jnp.arange(nspread)

    # X dimension
    i0_x = jnp.ceil(x_scaled - ns2).astype(jnp.int32)
    indices_x = (i0_x[:, None] + offsets[None, :]) % nf1
    z_x = (i0_x[:, None] + offsets[None, :]).astype(x_scaled.dtype) - x_scaled[:, None]
    weights_x = es_kernel(z_x, beta, c)

    # Y dimension
    i0_y = jnp.ceil(y_scaled - ns2).astype(jnp.int32)
    indices_y = (i0_y[:, None] + offsets[None, :]) % nf2
    z_y = (i0_y[:, None] + offsets[None, :]).astype(y_scaled.dtype) - y_scaled[:, None]
    weights_y = es_kernel(z_y, beta, c)

    return indices_x, indices_y, weights_x, weights_y


def spread_2d_impl(
    x: jax.Array,
    y: jax.Array,
    c: jax.Array,
    nf1: int,
    nf2: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    2D spreading implementation: scatter nonuniform values to grid.

    Args:
        x: Nonuniform x coordinates in [-pi, pi), shape (M,)
        y: Nonuniform y coordinates in [-pi, pi), shape (M,)
        c: Complex strengths, shape (M,) or (n_trans, M)
        nf1, nf2: Fine grid sizes
        kernel_params: Kernel parameters

    Returns:
        fw: Fine grid values, shape (nf1, nf2) or (n_trans, nf1, nf2)
    """
    c_flat, n_trans, is_batched = _prepare_batched_c(c)

    # Scale coordinates
    x_scaled = fold_rescale(x, nf1)
    y_scaled = fold_rescale(y, nf2)

    # Get kernel weights and indices for each dimension
    indices_x, indices_y, weights_x, weights_y = compute_kernel_weights_2d(x_scaled, y_scaled, nf1, nf2, kernel_params)

    # Initialize output grid
    fw = jnp.zeros((n_trans, nf1, nf2), dtype=c.dtype)

    # For 2D, we need to scatter to nspread x nspread points per nonuniform point
    # Compute 2D indices as linear indices into flattened grid
    # indices_2d[j, dy, dx] = indices_y[j, dy] * nf1 + indices_x[j, dx]
    indices_2d = indices_y[:, :, None] * nf1 + indices_x[:, None, :]  # (M, nspread, nspread)

    # Compute 2D weights as outer product
    # weights_2d[j, dy, dx] = weights_y[j, dy] * weights_x[j, dx]
    weights_2d = weights_y[:, :, None] * weights_x[:, None, :]  # (M, nspread, nspread)

    # Weighted contributions
    weighted_c = c_flat[:, :, None, None] * weights_2d[None, :, :, :]  # (n_trans, M, nspread, nspread)

    # Flatten for segment_sum
    indices_flat = indices_2d.ravel()  # (M * nspread * nspread,)
    weighted_c_flat = weighted_c.reshape(n_trans, -1)  # (n_trans, M * nspread * nspread)

    # Use segment_sum for efficient accumulation (faster than add.at)
    def segment_sum_for_one_transform(wc_t):
        return jax.ops.segment_sum(wc_t, indices_flat, num_segments=nf1 * nf2)

    fw_flat = jax.vmap(segment_sum_for_one_transform)(weighted_c_flat)
    fw = fw_flat.reshape(n_trans, nf1, nf2)

    if not is_batched:
        fw = fw[0]

    return fw


def interp_2d_impl(
    x: jax.Array,
    y: jax.Array,
    fw: jax.Array,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    2D interpolation implementation: gather grid values at nonuniform points.

    Args:
        x: Nonuniform x coordinates in [-pi, pi), shape (M,)
        y: Nonuniform y coordinates in [-pi, pi), shape (M,)
        fw: Fine grid values, shape (nf1, nf2) or (n_trans, nf1, nf2)
        kernel_params: Kernel parameters

    Returns:
        c: Interpolated values, shape (M,) or (n_trans, M)
    """
    fw_flat, nf1, nf2, _, is_batched = _prepare_batched_grid_2d(fw)
    M = x.shape[0]

    # Scale coordinates
    x_scaled = fold_rescale(x, nf1)
    y_scaled = fold_rescale(y, nf2)

    # Get kernel weights and indices
    indices_x, indices_y, weights_x, weights_y = compute_kernel_weights_2d(x_scaled, y_scaled, nf1, nf2, kernel_params)

    # Compute 2D indices and weights
    indices_2d = indices_y[:, :, None] * nf1 + indices_x[:, None, :]  # (M, nspread, nspread)
    weights_2d = weights_y[:, :, None] * weights_x[:, None, :]

    # Gather values
    # fw_gathered[t, j, dy, dx] = fw_flat[t, indices_2d[j, dy, dx]]
    indices_flat = indices_2d.ravel()
    fw_gathered = fw_flat[:, indices_flat].reshape(-1, M, kernel_params.nspread, kernel_params.nspread)

    # Apply weights and sum
    c = jnp.sum(fw_gathered * weights_2d[None, :, :, :], axis=(-2, -1))

    if not is_batched:
        c = c[0]

    return c


# ============================================================================
# 3D Spreading and Interpolation
# ============================================================================


def compute_kernel_weights_3d(
    x_scaled: jax.Array,
    y_scaled: jax.Array,
    z_scaled: jax.Array,
    nf1: int,
    nf2: int,
    nf3: int,
    kernel_params: KernelParams,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute kernel weights and grid indices for 3D spreading/interpolation.

    Args:
        x_scaled, y_scaled, z_scaled: Scaled coordinates, shape (M,) each
        nf1, nf2, nf3: Fine grid sizes
        kernel_params: Kernel parameters

    Returns:
        indices_x, indices_y, indices_z: Grid indices, shape (M, nspread) each
        weights_x, weights_y, weights_z: Kernel weights, shape (M, nspread) each
    """
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    c = kernel_params.c

    ns2 = nspread / 2.0
    offsets = jnp.arange(nspread)

    # X dimension
    i0_x = jnp.ceil(x_scaled - ns2).astype(jnp.int32)
    indices_x = (i0_x[:, None] + offsets[None, :]) % nf1
    z_x = (i0_x[:, None] + offsets[None, :]).astype(x_scaled.dtype) - x_scaled[:, None]
    weights_x = es_kernel(z_x, beta, c)

    # Y dimension
    i0_y = jnp.ceil(y_scaled - ns2).astype(jnp.int32)
    indices_y = (i0_y[:, None] + offsets[None, :]) % nf2
    z_y = (i0_y[:, None] + offsets[None, :]).astype(y_scaled.dtype) - y_scaled[:, None]
    weights_y = es_kernel(z_y, beta, c)

    # Z dimension
    i0_z = jnp.ceil(z_scaled - ns2).astype(jnp.int32)
    indices_z = (i0_z[:, None] + offsets[None, :]) % nf3
    z_z = (i0_z[:, None] + offsets[None, :]).astype(z_scaled.dtype) - z_scaled[:, None]
    weights_z = es_kernel(z_z, beta, c)

    return indices_x, indices_y, indices_z, weights_x, weights_y, weights_z


def spread_3d_impl(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    c: jax.Array,
    nf1: int,
    nf2: int,
    nf3: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    3D spreading implementation: scatter nonuniform values to grid.

    Args:
        x, y, z: Nonuniform coordinates in [-pi, pi), shape (M,) each
        c: Complex strengths, shape (M,) or (n_trans, M)
        nf1, nf2, nf3: Fine grid sizes
        kernel_params: Kernel parameters

    Returns:
        fw: Fine grid values, shape (nf1, nf2, nf3) or (n_trans, nf1, nf2, nf3)
    """
    c_flat, n_trans, is_batched = _prepare_batched_c(c)

    # Scale coordinates
    x_scaled = fold_rescale(x, nf1)
    y_scaled = fold_rescale(y, nf2)
    z_scaled = fold_rescale(z, nf3)

    # Get kernel weights and indices
    (indices_x, indices_y, indices_z, weights_x, weights_y, weights_z) = compute_kernel_weights_3d(
        x_scaled, y_scaled, z_scaled, nf1, nf2, nf3, kernel_params
    )

    # Compute 3D linear indices
    # indices_3d[j, dz, dy, dx] = indices_z[j,dz]*nf1*nf2 + indices_y[j,dy]*nf1 + indices_x[j,dx]
    indices_3d = (
        indices_z[:, :, None, None] * (nf1 * nf2) + indices_y[:, None, :, None] * nf1 + indices_x[:, None, None, :]
    )  # (M, nspread, nspread, nspread)

    # Compute 3D weights as outer product
    weights_3d = (
        weights_z[:, :, None, None] * weights_y[:, None, :, None] * weights_x[:, None, None, :]
    )  # (M, nspread, nspread, nspread)

    # Weighted contributions
    weighted_c = c_flat[:, :, None, None, None] * weights_3d[None, :, :, :, :]

    # Flatten for segment_sum
    indices_flat = indices_3d.ravel()
    weighted_c_flat = weighted_c.reshape(n_trans, -1)

    # Use segment_sum for efficient accumulation (faster than add.at)
    def segment_sum_for_one_transform(wc_t):
        return jax.ops.segment_sum(wc_t, indices_flat, num_segments=nf1 * nf2 * nf3)

    fw_flat = jax.vmap(segment_sum_for_one_transform)(weighted_c_flat)
    fw = fw_flat.reshape(n_trans, nf1, nf2, nf3)

    if not is_batched:
        fw = fw[0]

    return fw


def interp_3d_impl(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    fw: jax.Array,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    3D interpolation implementation: gather grid values at nonuniform points.

    Args:
        x, y, z: Nonuniform coordinates in [-pi, pi), shape (M,) each
        fw: Fine grid values, shape (nf1, nf2, nf3) or (n_trans, nf1, nf2, nf3)
        kernel_params: Kernel parameters

    Returns:
        c: Interpolated values, shape (M,) or (n_trans, M)
    """
    fw_flat, nf1, nf2, nf3, _, is_batched = _prepare_batched_grid_3d(fw)
    M = x.shape[0]
    nspread = kernel_params.nspread

    # Scale coordinates
    x_scaled = fold_rescale(x, nf1)
    y_scaled = fold_rescale(y, nf2)
    z_scaled = fold_rescale(z, nf3)

    # Get kernel weights and indices
    (indices_x, indices_y, indices_z, weights_x, weights_y, weights_z) = compute_kernel_weights_3d(
        x_scaled, y_scaled, z_scaled, nf1, nf2, nf3, kernel_params
    )

    # Compute 3D indices and weights
    indices_3d = (
        indices_z[:, :, None, None] * (nf1 * nf2) + indices_y[:, None, :, None] * nf1 + indices_x[:, None, None, :]
    )
    weights_3d = weights_z[:, :, None, None] * weights_y[:, None, :, None] * weights_x[:, None, None, :]

    # Gather values
    indices_flat = indices_3d.ravel()
    fw_gathered = fw_flat[:, indices_flat].reshape(-1, M, nspread, nspread, nspread)

    # Apply weights and sum
    c = jnp.sum(fw_gathered * weights_3d[None, :, :, :, :], axis=(-3, -2, -1))

    if not is_batched:
        c = c[0]

    return c


# ============================================================================
# Public API with Custom VJP
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def spread_1d(
    x: jax.Array,
    c: jax.Array,
    nf: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    Spread nonuniform point values to a 1D uniform grid.

    Type 1 NUFFT spreading operation:
        fw[k] = sum_j c[j] * phi((k - x[j] * nf / (2*pi)) / w)

    where phi is the ES kernel with width w = nspread.

    Args:
        x: Nonuniform point coordinates in [-pi, pi), shape (M,)
        c: Complex strengths at nonuniform points, shape (M,) or (n_trans, M)
        nf: Fine grid size
        kernel_params: Kernel parameters (nspread, beta, c, upsampfac)

    Returns:
        fw: Fine grid values, shape (nf,) or (n_trans, nf)
    """
    return spread_1d_impl(x, c, nf, kernel_params)


def spread_1d_fwd(x, c, nf, kernel_params):
    """Forward pass for spread_1d VJP."""
    result = spread_1d_impl(x, c, nf, kernel_params)
    return result, (x, c)


def spread_1d_bwd(nf, kernel_params, res, g):
    """Backward pass for spread_1d VJP.

    The adjoint of spreading w.r.t. c is interpolation.
    The adjoint w.r.t. x requires kernel derivative.
    """
    x, c = res

    # Gradient w.r.t. c: adjoint of spreading is interpolation
    dc = interp_1d_impl(x, g, kernel_params)

    # Gradient w.r.t. x: requires kernel derivative
    # d/dx[spread(x,c)] involves dphi/dx
    dx = _spread_1d_grad_x(x, c, g, nf, kernel_params)

    return (dx, dc)


spread_1d.defvjp(spread_1d_fwd, spread_1d_bwd)


def _spread_1d_grad_x(
    x: jax.Array,
    c: jax.Array,
    g: jax.Array,
    nf: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    Compute gradient of spread_1d with respect to x.

    The gradient involves the kernel derivative:
        dx[j] = sum_k g[k] * c[j] * dphi/dx(k - x[j] * nf / (2*pi))
    """
    c_flat, _, _ = _prepare_batched_c(c)
    g_flat, _, _, _ = _prepare_batched_grid_1d(g)

    # Scale coordinates
    x_scaled = fold_rescale(x, nf)

    # Get indices, weights, and weight derivatives
    indices, weights, dweights = compute_kernel_weights_derivative_1d(x_scaled, nf, kernel_params)

    # Gather g values at kernel support points
    # g_gathered[t, j, k] = g[t, indices[j, k]]
    g_gathered = g_flat[:, indices]  # (n_trans, M, nspread)

    # Compute gradient: sum over transforms and kernel support
    # dx[j] = sum_t sum_k real(conj(c[t,j]) * g[t, indices[j,k]] * dweights[j,k])
    # Note: for complex c, gradient is w.r.t. real-valued x
    # The contribution is: Re(conj(c) * g * dphi)
    contrib = g_gathered * dweights[None, :, :]  # (n_trans, M, nspread)
    # Sum over kernel support
    contrib_sum = jnp.sum(contrib, axis=-1)  # (n_trans, M)
    # Multiply by c and take real part (since x is real)
    dx_per_trans = jnp.real(jnp.conj(c_flat) * contrib_sum)  # (n_trans, M)
    # Sum over transforms
    dx = jnp.sum(dx_per_trans, axis=0)  # (M,)

    return dx


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def interp_1d(
    x: jax.Array,
    fw: jax.Array,
    nf: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    Interpolate from 1D uniform grid to nonuniform points.

    Type 2 NUFFT interpolation operation:
        c[j] = sum_k fw[k] * phi((k - x[j] * nf / (2*pi)) / w)

    Args:
        x: Nonuniform point coordinates in [-pi, pi), shape (M,)
        fw: Fine grid values, shape (nf,) or (n_trans, nf)
        nf: Fine grid size (must match fw)
        kernel_params: Kernel parameters

    Returns:
        c: Interpolated values, shape (M,) or (n_trans, M)
    """
    return interp_1d_impl(x, fw, kernel_params)


def interp_1d_fwd(x, fw, nf, kernel_params):
    """Forward pass for interp_1d VJP."""
    result = interp_1d_impl(x, fw, kernel_params)
    return result, (x, fw)


def interp_1d_bwd(nf, kernel_params, res, g):
    """Backward pass for interp_1d VJP.

    The adjoint of interpolation w.r.t. fw is spreading.
    """
    x, fw = res

    # Gradient w.r.t. fw: adjoint of interpolation is spreading
    dfw = spread_1d_impl(x, g, nf, kernel_params)

    # Gradient w.r.t. x: similar to spread gradient
    dx = _interp_1d_grad_x(x, fw, g, nf, kernel_params)

    return (dx, dfw)


interp_1d.defvjp(interp_1d_fwd, interp_1d_bwd)


def _interp_1d_grad_x(
    x: jax.Array,
    fw: jax.Array,
    g: jax.Array,
    nf: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    Compute gradient of interp_1d with respect to x.

    The gradient involves the kernel derivative:
        dx[j] = sum_k fw[k] * dphi/dx(k - x[j] * nf / (2*pi)) * g[j]
    """
    fw_flat, _, _, _ = _prepare_batched_grid_1d(fw)
    g_flat, _, _ = _prepare_batched_c(g)

    # Scale coordinates
    x_scaled = fold_rescale(x, nf)

    # Get indices, weights, and weight derivatives
    indices, weights, dweights = compute_kernel_weights_derivative_1d(x_scaled, nf, kernel_params)

    # Gather fw values at kernel support points
    fw_gathered = fw_flat[:, indices]  # (n_trans, M, nspread)

    # Compute derivative of c w.r.t. x
    # dc/dx[j] = sum_k fw[k] * dweights[j, k]
    dc_dx = jnp.sum(fw_gathered * dweights[None, :, :], axis=-1)  # (n_trans, M)

    # Chain rule with g
    dx_per_trans = jnp.real(jnp.conj(g_flat) * dc_dx)  # (n_trans, M)
    dx = jnp.sum(dx_per_trans, axis=0)  # (M,)

    return dx


# ============================================================================
# 2D Public API with Custom VJP
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def spread_2d(
    x: jax.Array,
    y: jax.Array,
    c: jax.Array,
    nf1: int,
    nf2: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    Spread nonuniform point values to a 2D uniform grid.

    Args:
        x: Nonuniform x coordinates in [-pi, pi), shape (M,)
        y: Nonuniform y coordinates in [-pi, pi), shape (M,)
        c: Complex strengths, shape (M,) or (n_trans, M)
        nf1, nf2: Fine grid sizes
        kernel_params: Kernel parameters

    Returns:
        fw: Fine grid values, shape (nf1, nf2) or (n_trans, nf1, nf2)
    """
    return spread_2d_impl(x, y, c, nf1, nf2, kernel_params)


def spread_2d_fwd(x, y, c, nf1, nf2, kernel_params):
    result = spread_2d_impl(x, y, c, nf1, nf2, kernel_params)
    return result, (x, y, c)


def spread_2d_bwd(nf1, nf2, kernel_params, res, g):
    x, y, c = res

    # Gradient w.r.t. c
    dc = interp_2d_impl(x, y, g, kernel_params)

    # Gradient w.r.t. x and y
    dx, dy = _spread_2d_grad_xy(x, y, c, g, nf1, nf2, kernel_params)

    return (dx, dy, dc)


spread_2d.defvjp(spread_2d_fwd, spread_2d_bwd)


def _spread_2d_grad_xy(x, y, c, g, nf1, nf2, kernel_params):
    """Compute gradients of spread_2d w.r.t. x and y."""
    c_flat, _, _ = _prepare_batched_c(c)
    g_flat, _, _, _, _ = _prepare_batched_grid_2d(g)
    M = x.shape[0]
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    kc = kernel_params.c

    # Scale coordinates
    x_scaled = fold_rescale(x, nf1)
    y_scaled = fold_rescale(y, nf2)

    ns2 = nspread / 2.0
    offsets = jnp.arange(nspread)
    scale_x = nf1 / (2.0 * jnp.pi)
    scale_y = nf2 / (2.0 * jnp.pi)

    # X dimension (fused kernel + derivative computation)
    i0_x = jnp.ceil(x_scaled - ns2).astype(jnp.int32)
    indices_x = (i0_x[:, None] + offsets[None, :]) % nf1
    z_x = (i0_x[:, None] + offsets[None, :]).astype(x.dtype) - x_scaled[:, None]
    weights_x, dweights_x_raw = es_kernel_with_derivative(z_x, beta, kc)
    dweights_x = -dweights_x_raw * scale_x

    # Y dimension (fused kernel + derivative computation)
    i0_y = jnp.ceil(y_scaled - ns2).astype(jnp.int32)
    indices_y = (i0_y[:, None] + offsets[None, :]) % nf2
    z_y = (i0_y[:, None] + offsets[None, :]).astype(y.dtype) - y_scaled[:, None]
    weights_y, dweights_y_raw = es_kernel_with_derivative(z_y, beta, kc)
    dweights_y = -dweights_y_raw * scale_y

    # 2D indices
    indices_2d = indices_y[:, :, None] * nf1 + indices_x[:, None, :]
    indices_flat = indices_2d.ravel()

    # Gather g values
    g_gathered = g_flat[:, indices_flat].reshape(-1, M, nspread, nspread)

    # For dx: use dweights_x, weights_y
    weights_2d_dx = weights_y[:, :, None] * dweights_x[:, None, :]
    contrib_dx = jnp.sum(g_gathered * weights_2d_dx[None, :, :, :], axis=(-2, -1))
    dx_per_trans = jnp.real(jnp.conj(c_flat) * contrib_dx)
    dx = jnp.sum(dx_per_trans, axis=0)

    # For dy: use weights_x, dweights_y
    weights_2d_dy = dweights_y[:, :, None] * weights_x[:, None, :]
    contrib_dy = jnp.sum(g_gathered * weights_2d_dy[None, :, :, :], axis=(-2, -1))
    dy_per_trans = jnp.real(jnp.conj(c_flat) * contrib_dy)
    dy = jnp.sum(dy_per_trans, axis=0)

    return dx, dy


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def interp_2d(
    x: jax.Array,
    y: jax.Array,
    fw: jax.Array,
    nf1: int,
    nf2: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    Interpolate from 2D uniform grid to nonuniform points.

    Args:
        x: Nonuniform x coordinates in [-pi, pi), shape (M,)
        y: Nonuniform y coordinates in [-pi, pi), shape (M,)
        fw: Fine grid values, shape (nf1, nf2) or (n_trans, nf1, nf2)
        nf1, nf2: Fine grid sizes
        kernel_params: Kernel parameters

    Returns:
        c: Interpolated values, shape (M,) or (n_trans, M)
    """
    return interp_2d_impl(x, y, fw, kernel_params)


def interp_2d_fwd(x, y, fw, nf1, nf2, kernel_params):
    result = interp_2d_impl(x, y, fw, kernel_params)
    return result, (x, y, fw)


def interp_2d_bwd(nf1, nf2, kernel_params, res, g):
    x, y, fw = res

    # Gradient w.r.t. fw
    dfw = spread_2d_impl(x, y, g, nf1, nf2, kernel_params)

    # Gradient w.r.t. x and y
    dx, dy = _interp_2d_grad_xy(x, y, fw, g, nf1, nf2, kernel_params)

    return (dx, dy, dfw)


interp_2d.defvjp(interp_2d_fwd, interp_2d_bwd)


def _interp_2d_grad_xy(x, y, fw, g, nf1, nf2, kernel_params):
    """Compute gradients of interp_2d w.r.t. x and y."""
    fw_flat, _, _, _, _ = _prepare_batched_grid_2d(fw)
    g_flat, _, _ = _prepare_batched_c(g)
    M = x.shape[0]
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    kc = kernel_params.c

    x_scaled = fold_rescale(x, nf1)
    y_scaled = fold_rescale(y, nf2)

    ns2 = nspread / 2.0
    offsets = jnp.arange(nspread)
    scale_x = nf1 / (2.0 * jnp.pi)
    scale_y = nf2 / (2.0 * jnp.pi)

    # X dimension (fused kernel + derivative computation)
    i0_x = jnp.ceil(x_scaled - ns2).astype(jnp.int32)
    indices_x = (i0_x[:, None] + offsets[None, :]) % nf1
    z_x = (i0_x[:, None] + offsets[None, :]).astype(x.dtype) - x_scaled[:, None]
    weights_x, dweights_x_raw = es_kernel_with_derivative(z_x, beta, kc)
    dweights_x = -dweights_x_raw * scale_x

    # Y dimension (fused kernel + derivative computation)
    i0_y = jnp.ceil(y_scaled - ns2).astype(jnp.int32)
    indices_y = (i0_y[:, None] + offsets[None, :]) % nf2
    z_y = (i0_y[:, None] + offsets[None, :]).astype(y.dtype) - y_scaled[:, None]
    weights_y, dweights_y_raw = es_kernel_with_derivative(z_y, beta, kc)
    dweights_y = -dweights_y_raw * scale_y

    # 2D indices
    indices_2d = indices_y[:, :, None] * nf1 + indices_x[:, None, :]
    indices_flat = indices_2d.ravel()

    # Gather fw values
    fw_gathered = fw_flat[:, indices_flat].reshape(-1, M, nspread, nspread)

    # For dx
    weights_2d_dx = weights_y[:, :, None] * dweights_x[:, None, :]
    dc_dx = jnp.sum(fw_gathered * weights_2d_dx[None, :, :, :], axis=(-2, -1))
    dx_per_trans = jnp.real(jnp.conj(g_flat) * dc_dx)
    dx = jnp.sum(dx_per_trans, axis=0)

    # For dy
    weights_2d_dy = dweights_y[:, :, None] * weights_x[:, None, :]
    dc_dy = jnp.sum(fw_gathered * weights_2d_dy[None, :, :, :], axis=(-2, -1))
    dy_per_trans = jnp.real(jnp.conj(g_flat) * dc_dy)
    dy = jnp.sum(dy_per_trans, axis=0)

    return dx, dy


# ============================================================================
# 3D Public API with Custom VJP
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def spread_3d(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    c: jax.Array,
    nf1: int,
    nf2: int,
    nf3: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    Spread nonuniform point values to a 3D uniform grid.

    Args:
        x, y, z: Nonuniform coordinates in [-pi, pi), shape (M,) each
        c: Complex strengths, shape (M,) or (n_trans, M)
        nf1, nf2, nf3: Fine grid sizes
        kernel_params: Kernel parameters

    Returns:
        fw: Fine grid values, shape (nf1, nf2, nf3) or (n_trans, nf1, nf2, nf3)
    """
    return spread_3d_impl(x, y, z, c, nf1, nf2, nf3, kernel_params)


def spread_3d_fwd(x, y, z, c, nf1, nf2, nf3, kernel_params):
    result = spread_3d_impl(x, y, z, c, nf1, nf2, nf3, kernel_params)
    return result, (x, y, z, c)


def spread_3d_bwd(nf1, nf2, nf3, kernel_params, res, g):
    x, y, z, c = res

    # Gradient w.r.t. c
    dc = interp_3d_impl(x, y, z, g, kernel_params)

    # Gradient w.r.t. x, y, z
    dx, dy, dz = _spread_3d_grad_xyz(x, y, z, c, g, nf1, nf2, nf3, kernel_params)

    return (dx, dy, dz, dc)


spread_3d.defvjp(spread_3d_fwd, spread_3d_bwd)


def _spread_3d_grad_xyz(x, y, z, c, g, nf1, nf2, nf3, kernel_params):
    """Compute gradients of spread_3d w.r.t. x, y, z."""
    c_flat, _, _ = _prepare_batched_c(c)
    g_flat, _, _, _, _, _ = _prepare_batched_grid_3d(g)
    M = x.shape[0]
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    kc = kernel_params.c

    x_scaled = fold_rescale(x, nf1)
    y_scaled = fold_rescale(y, nf2)
    z_scaled = fold_rescale(z, nf3)

    ns2 = nspread / 2.0
    offsets = jnp.arange(nspread)
    scale_x = nf1 / (2.0 * jnp.pi)
    scale_y = nf2 / (2.0 * jnp.pi)
    scale_z = nf3 / (2.0 * jnp.pi)

    # X dimension (fused kernel + derivative computation)
    i0_x = jnp.ceil(x_scaled - ns2).astype(jnp.int32)
    indices_x = (i0_x[:, None] + offsets[None, :]) % nf1
    z_x = (i0_x[:, None] + offsets[None, :]).astype(x.dtype) - x_scaled[:, None]
    weights_x, dweights_x_raw = es_kernel_with_derivative(z_x, beta, kc)
    dweights_x = -dweights_x_raw * scale_x

    # Y dimension (fused kernel + derivative computation)
    i0_y = jnp.ceil(y_scaled - ns2).astype(jnp.int32)
    indices_y = (i0_y[:, None] + offsets[None, :]) % nf2
    z_y = (i0_y[:, None] + offsets[None, :]).astype(y.dtype) - y_scaled[:, None]
    weights_y, dweights_y_raw = es_kernel_with_derivative(z_y, beta, kc)
    dweights_y = -dweights_y_raw * scale_y

    # Z dimension (fused kernel + derivative computation)
    i0_z = jnp.ceil(z_scaled - ns2).astype(jnp.int32)
    indices_z = (i0_z[:, None] + offsets[None, :]) % nf3
    z_z = (i0_z[:, None] + offsets[None, :]).astype(z.dtype) - z_scaled[:, None]
    weights_z, dweights_z_raw = es_kernel_with_derivative(z_z, beta, kc)
    dweights_z = -dweights_z_raw * scale_z

    # 3D indices
    indices_3d = (
        indices_z[:, :, None, None] * (nf1 * nf2) + indices_y[:, None, :, None] * nf1 + indices_x[:, None, None, :]
    )
    indices_flat = indices_3d.ravel()

    # Gather g values
    g_gathered = g_flat[:, indices_flat].reshape(-1, M, nspread, nspread, nspread)

    # For dx
    weights_3d_dx = weights_z[:, :, None, None] * weights_y[:, None, :, None] * dweights_x[:, None, None, :]
    contrib_dx = jnp.sum(g_gathered * weights_3d_dx[None, :, :, :, :], axis=(-3, -2, -1))
    dx_per_trans = jnp.real(jnp.conj(c_flat) * contrib_dx)
    dx = jnp.sum(dx_per_trans, axis=0)

    # For dy
    weights_3d_dy = weights_z[:, :, None, None] * dweights_y[:, None, :, None] * weights_x[:, None, None, :]
    contrib_dy = jnp.sum(g_gathered * weights_3d_dy[None, :, :, :, :], axis=(-3, -2, -1))
    dy_per_trans = jnp.real(jnp.conj(c_flat) * contrib_dy)
    dy = jnp.sum(dy_per_trans, axis=0)

    # For dz
    weights_3d_dz = dweights_z[:, :, None, None] * weights_y[:, None, :, None] * weights_x[:, None, None, :]
    contrib_dz = jnp.sum(g_gathered * weights_3d_dz[None, :, :, :, :], axis=(-3, -2, -1))
    dz_per_trans = jnp.real(jnp.conj(c_flat) * contrib_dz)
    dz = jnp.sum(dz_per_trans, axis=0)

    return dx, dy, dz


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def interp_3d(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    fw: jax.Array,
    nf1: int,
    nf2: int,
    nf3: int,
    kernel_params: KernelParams,
) -> jax.Array:
    """
    Interpolate from 3D uniform grid to nonuniform points.

    Args:
        x, y, z: Nonuniform coordinates in [-pi, pi), shape (M,) each
        fw: Fine grid values, shape (nf1, nf2, nf3) or (n_trans, nf1, nf2, nf3)
        nf1, nf2, nf3: Fine grid sizes
        kernel_params: Kernel parameters

    Returns:
        c: Interpolated values, shape (M,) or (n_trans, M)
    """
    return interp_3d_impl(x, y, z, fw, kernel_params)


def interp_3d_fwd(x, y, z, fw, nf1, nf2, nf3, kernel_params):
    result = interp_3d_impl(x, y, z, fw, kernel_params)
    return result, (x, y, z, fw)


def interp_3d_bwd(nf1, nf2, nf3, kernel_params, res, g):
    x, y, z, fw = res

    # Gradient w.r.t. fw
    dfw = spread_3d_impl(x, y, z, g, nf1, nf2, nf3, kernel_params)

    # Gradient w.r.t. x, y, z
    dx, dy, dz = _interp_3d_grad_xyz(x, y, z, fw, g, nf1, nf2, nf3, kernel_params)

    return (dx, dy, dz, dfw)


interp_3d.defvjp(interp_3d_fwd, interp_3d_bwd)


def _interp_3d_grad_xyz(x, y, z, fw, g, nf1, nf2, nf3, kernel_params):
    """Compute gradients of interp_3d w.r.t. x, y, z."""
    fw_flat, _, _, _, _, _ = _prepare_batched_grid_3d(fw)
    g_flat, _, _ = _prepare_batched_c(g)
    M = x.shape[0]
    nspread = kernel_params.nspread
    beta = kernel_params.beta
    kc = kernel_params.c

    x_scaled = fold_rescale(x, nf1)
    y_scaled = fold_rescale(y, nf2)
    z_scaled = fold_rescale(z, nf3)

    ns2 = nspread / 2.0
    offsets = jnp.arange(nspread)
    scale_x = nf1 / (2.0 * jnp.pi)
    scale_y = nf2 / (2.0 * jnp.pi)
    scale_z = nf3 / (2.0 * jnp.pi)

    # X dimension (fused kernel + derivative computation)
    i0_x = jnp.ceil(x_scaled - ns2).astype(jnp.int32)
    indices_x = (i0_x[:, None] + offsets[None, :]) % nf1
    z_x = (i0_x[:, None] + offsets[None, :]).astype(x.dtype) - x_scaled[:, None]
    weights_x, dweights_x_raw = es_kernel_with_derivative(z_x, beta, kc)
    dweights_x = -dweights_x_raw * scale_x

    # Y dimension (fused kernel + derivative computation)
    i0_y = jnp.ceil(y_scaled - ns2).astype(jnp.int32)
    indices_y = (i0_y[:, None] + offsets[None, :]) % nf2
    z_y = (i0_y[:, None] + offsets[None, :]).astype(y.dtype) - y_scaled[:, None]
    weights_y, dweights_y_raw = es_kernel_with_derivative(z_y, beta, kc)
    dweights_y = -dweights_y_raw * scale_y

    # Z dimension (fused kernel + derivative computation)
    i0_z = jnp.ceil(z_scaled - ns2).astype(jnp.int32)
    indices_z = (i0_z[:, None] + offsets[None, :]) % nf3
    z_z = (i0_z[:, None] + offsets[None, :]).astype(z.dtype) - z_scaled[:, None]
    weights_z, dweights_z_raw = es_kernel_with_derivative(z_z, beta, kc)
    dweights_z = -dweights_z_raw * scale_z

    # 3D indices
    indices_3d = (
        indices_z[:, :, None, None] * (nf1 * nf2) + indices_y[:, None, :, None] * nf1 + indices_x[:, None, None, :]
    )
    indices_flat = indices_3d.ravel()

    # Gather fw values
    fw_gathered = fw_flat[:, indices_flat].reshape(-1, M, nspread, nspread, nspread)

    # For dx
    weights_3d_dx = weights_z[:, :, None, None] * weights_y[:, None, :, None] * dweights_x[:, None, None, :]
    dc_dx = jnp.sum(fw_gathered * weights_3d_dx[None, :, :, :, :], axis=(-3, -2, -1))
    dx_per_trans = jnp.real(jnp.conj(g_flat) * dc_dx)
    dx = jnp.sum(dx_per_trans, axis=0)

    # For dy
    weights_3d_dy = weights_z[:, :, None, None] * dweights_y[:, None, :, None] * weights_x[:, None, None, :]
    dc_dy = jnp.sum(fw_gathered * weights_3d_dy[None, :, :, :, :], axis=(-3, -2, -1))
    dy_per_trans = jnp.real(jnp.conj(g_flat) * dc_dy)
    dy = jnp.sum(dy_per_trans, axis=0)

    # For dz
    weights_3d_dz = dweights_z[:, :, None, None] * weights_y[:, None, :, None] * weights_x[:, None, None, :]
    dc_dz = jnp.sum(fw_gathered * weights_3d_dz[None, :, :, :, :], axis=(-3, -2, -1))
    dz_per_trans = jnp.real(jnp.conj(g_flat) * dc_dz)
    dz = jnp.sum(dz_per_trans, axis=0)

    return dx, dy, dz
