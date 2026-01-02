# nufftax

**Pure JAX implementation of the Non-Uniform Fast Fourier Transform (NUFFT).**

[![CI](https://github.com/geoffroyO/nufftax/actions/workflows/ci.yml/badge.svg)](https://github.com/geoffroyO/nufftax/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

**nufftax** provides fully differentiable NUFFT operations in pure JAX. No C++ bindings or XLA custom calls - just JAX operations that work seamlessly with `jit`, `grad`, `vmap`, `jvp`, and `vjp`.

## Features

- **Pure JAX implementation** - Full compatibility with JAX transformations
- **All three NUFFT types**:
  - Type 1: nonuniform to uniform (spreading)
  - Type 2: uniform to nonuniform (interpolation)
  - Type 3: nonuniform to nonuniform
- **1D, 2D, and 3D** transforms
- **Differentiable** with respect to both point coordinates and values
- **GPU acceleration** - Runs on CPU and GPU without code changes
- **Configurable precision** - From 1e-2 to 1e-14

## Installation

```bash
pip install nufftax
```

From source:

```bash
git clone https://github.com/geoffroyO/nufftax.git
cd nufftax
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from nufftax import nufft1d1, nufft1d2

# Nonuniform points in [-pi, pi)
x = jnp.array([0.1, 0.5, 1.0, 2.0, -1.5])
c = jnp.array([1+1j, 2-1j, 0.5, 1j, -1+0.5j])

# Type 1: nonuniform points -> Fourier modes
f = nufft1d1(x, c, n_modes=64, eps=1e-6)

# Type 2: Fourier modes -> nonuniform points
c_interp = nufft1d2(x, f, eps=1e-6)
```

## Autodifferentiation

Gradients work out of the box:

```python
import jax

# Gradient w.r.t. strengths
def loss_c(c):
    f = nufft1d1(x, c, n_modes=64, eps=1e-6)
    return jnp.sum(jnp.abs(f) ** 2)

grad_c = jax.grad(loss_c)(c)

# Gradient w.r.t. point coordinates
def loss_x(x):
    f = nufft1d1(x, c, n_modes=64, eps=1e-6)
    return jnp.sum(jnp.abs(f) ** 2)

grad_x = jax.grad(loss_x)(x)

# Batched transforms
batched_nufft = jax.vmap(lambda xi: nufft1d1(xi, c, n_modes=64))
x_batch = jnp.stack([x, x + 0.1, x + 0.2])
f_batch = batched_nufft(x_batch)  # Shape: (3, 64)
```

## API Reference

### Type 1: Nonuniform to Uniform

Computes: `f[k] = sum_j c[j] * exp(i * isign * k * x[j])`

```python
from nufftax import nufft1d1, nufft2d1, nufft3d1

f = nufft1d1(x, c, n_modes, eps=1e-6, isign=1)
f = nufft2d1(x, y, c, n_modes, eps=1e-6, isign=1)
f = nufft3d1(x, y, z, c, n_modes, eps=1e-6, isign=1)
```

### Type 2: Uniform to Nonuniform

Computes: `c[j] = sum_k f[k] * exp(i * isign * k * x[j])`

```python
from nufftax import nufft1d2, nufft2d2, nufft3d2

c = nufft1d2(x, f, eps=1e-6, isign=1)
c = nufft2d2(x, y, f, eps=1e-6, isign=1)
c = nufft3d2(x, y, z, f, eps=1e-6, isign=1)
```

### Type 3: Nonuniform to Nonuniform

Computes: `f[k] = sum_j c[j] * exp(i * isign * s[k] * x[j])`

```python
from nufftax import nufft1d3, nufft2d3, nufft3d3
from nufftax import compute_type3_grid_size

# Compute grid size from data bounds (required for JIT)
n_modes = compute_type3_grid_size(x_extent, s_extent, eps=1e-6)

f = nufft1d3(x, c, s, n_modes, eps=1e-6, isign=1)
f = nufft2d3(x, y, c, s, t, n_modes, eps=1e-6, isign=1)
f = nufft3d3(x, y, z, c, s, t, u, n_modes, eps=1e-6, isign=1)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `x, y, z` | Nonuniform source points in [-pi, pi) |
| `s, t, u` | Nonuniform target frequencies (Type 3 only) |
| `c` | Complex strengths at source points |
| `f` | Fourier mode coefficients |
| `n_modes` | Number of output modes (int or tuple) |
| `eps` | Requested precision (1e-2 to 1e-14) |
| `isign` | Sign of exponent: +1 or -1 |

## Algorithm

nufftax implements the NUFFT using:

1. **Spreading/Interpolation** - Convolution with the exponential of semicircle (ES) kernel
2. **FFT** - Standard FFT on an oversampled grid (2x by default)
3. **Deconvolution** - Division by kernel Fourier coefficients

The ES kernel provides near-optimal accuracy for a given support width. All operations are implemented in pure JAX, enabling automatic differentiation through the entire transform.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT

## Acknowledgments

Algorithm based on [FINUFFT](https://github.com/flatironinstitute/finufft) by the Flatiron Institute.
