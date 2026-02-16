"""
nufftax: Pure JAX implementation of Non-Uniform FFT

This package provides NUFFT (Non-Uniform Fast Fourier Transform) operations
that are fully compatible with JAX transformations including:
- jit: JIT compilation
- grad/vjp: Reverse-mode automatic differentiation
- jvp: Forward-mode automatic differentiation
- vmap: Automatic vectorization

JAX Transformation Support
--------------------------
+------------------+-----+----------+-----+------+
| Transform        | jit | grad/vjp | jvp | vmap |
+------------------+-----+----------+-----+------+
| Type 1 (1D/2D/3D)|  Y  |    Y     |  Y  |  Y   |
| Type 2 (1D/2D/3D)|  Y  |    Y     |  Y  |  Y   |
| Type 3 (1D/2D/3D)|  Y  |    Y     |  Y  |  Y   |
+------------------+-----+----------+-----+------+

Differentiable inputs:
- Type 1: grad w.r.t. c (strengths) and x, y, z (coordinates)
- Type 2: grad w.r.t. f (Fourier modes) and x, y, z (coordinates)
- Type 3: grad w.r.t. c (strengths), x, y, z (source coordinates), and s, t, u (target frequencies)

Main functions:
- nufft1d1, nufft1d2: 1D NUFFT Type 1 and Type 2
- nufft2d1, nufft2d2: 2D NUFFT Type 1 and Type 2
- nufft3d1, nufft3d2: 3D NUFFT Type 1 and Type 2
- nufft1d3, nufft2d3, nufft3d3: Type 3 (nonuniform to nonuniform)

Example:
    import jax
    import jax.numpy as jnp
    from nufftax import nufft1d1, nufft1d2

    # Type 1: Nonuniform points to uniform Fourier modes
    x = jnp.array([0.1, 0.5, 1.0, 2.0])  # Nonuniform points
    c = jnp.array([1+1j, 2-1j, 0.5, 1j])  # Complex strengths
    f = nufft1d1(x, c, n_modes=64, eps=1e-6)

    # Type 2: Uniform Fourier modes to nonuniform points
    c2 = nufft1d2(x, f, eps=1e-6)

    # Gradients work directly with jax.grad/jax.vjp/jax.jvp
    grad_c = jax.grad(lambda c: jnp.sum(jnp.abs(nufft1d1(x, c, 64))**2))(c)
    grad_x = jax.grad(lambda x: jnp.sum(jnp.abs(nufft1d1(x, c, 64))**2))(x)
"""

__version__ = "0.2.1"

# Type 1 transforms (Nonuniform to Uniform) - with autodiff support
# Type 2 transforms (Uniform to Nonuniform) - with autodiff support
# Type 3 transforms (Nonuniform to Nonuniform) - with autodiff support
from .transforms.autodiff import (
    nufft1d1,
    nufft1d2,
    nufft1d3,
    nufft2d1,
    nufft2d2,
    nufft2d3,
    nufft3d1,
    nufft3d2,
    nufft3d3,
)

# Type 3 grid size helpers
from .transforms.nufft3 import (
    compute_type3_grid_size,
    compute_type3_grid_sizes_2d,
    compute_type3_grid_sizes_3d,
)

# Centered FFT utilities
from .utils.fft import fftc, from_complex, ifftc, to_complex

__all__ = [
    # Type 1 transforms
    "nufft1d1",
    "nufft2d1",
    "nufft3d1",
    # Type 2 transforms
    "nufft1d2",
    "nufft2d2",
    "nufft3d2",
    # Type 3 transforms
    "nufft1d3",
    "nufft2d3",
    "nufft3d3",
    # Type 3 JIT helpers
    "compute_type3_grid_size",
    "compute_type3_grid_sizes_2d",
    "compute_type3_grid_sizes_3d",
    # Centered FFT utilities
    "fftc",
    "ifftc",
    "to_complex",
    "from_complex",
]
