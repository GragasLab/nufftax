# nufftax 🌊

**Non-Uniform FFTs that actually play nice with JAX.** ✨

[![CI](https://github.com/geoffroyO/nufftax/actions/workflows/ci.yml/badge.svg)](https://github.com/geoffroyO/nufftax/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Ever tried to `jax.grad` through a NUFFT and got hit with a wall of errors? 😤 Yeah, us too.

**nufftax** is a pure JAX implementation of the Non-Uniform Fast Fourier Transform. No C++ bindings, no XLA custom calls, no tears. Just JAX all the way down. 🐍

## ✨ Features

- 🧬 **Pure JAX** - Works with `jit`, `grad`, `vmap`, `jvp`, `vjp`... the whole gang
- 📐 **1D, 2D, 3D** - Type 1 (nonuniform → uniform) and Type 2 (uniform → nonuniform)
- 🎯 **Differentiable** - Gradients with respect to both points and values
- 🚀 **GPU-ready** - Runs on CPU and GPU without code changes

## 📦 Installation

```bash
pip install nufftax
```

Or from source:

```bash
git clone https://github.com/geoffroyO/nufftax.git
cd nufftax
pip install -e ".[dev]"
```

## 🏁 Quick Start

```python
import jax.numpy as jnp
from nufftax import nufft1d1, nufft1d2

# Some nonuniform points in [0, 2π)
x = jnp.array([0.1, 0.5, 1.0, 2.0, 4.5])
c = jnp.array([1+1j, 2-1j, 0.5, 1j, -1+0.5j])

# Type 1: Nonuniform points → Fourier modes
f = nufft1d1(x, c, n_modes=64, eps=1e-6)

# Type 2: Fourier modes → Nonuniform points
c_back = nufft1d2(x, f, eps=1e-6)
```

## 🎉 The Fun Part: Autodiff Just Works

```python
import jax

# Gradient w.r.t. the strengths
def loss(c):
    f = nufft1d1(x, c, n_modes=64, eps=1e-6)
    return jnp.sum(jnp.abs(f) ** 2)

grad_c = jax.grad(loss)(c)  # It just works! 🎊

# Gradient w.r.t. the points (yes, really)
def loss_x(x):
    f = nufft1d1(x, c, n_modes=64, eps=1e-6)
    return jnp.sum(jnp.abs(f) ** 2)

grad_x = jax.grad(loss_x)(x)  # This too! 🤯

# Batch over multiple sets of points
batched_nufft = jax.vmap(lambda xi: nufft1d1(xi, c, n_modes=64))
x_batch = jnp.stack([x, x + 0.1, x + 0.2])
f_batch = batched_nufft(x_batch)  # Shape: (3, 64) 📦
```

## 📚 API Reference

### Type 1: Nonuniform → Uniform 🔀

```python
# 1D
f = nufft1d1(x, c, n_modes, eps=1e-6, iflag=1)

# 2D
f = nufft2d1(x, y, c, n_modes, eps=1e-6, iflag=1)

# 3D
f = nufft3d1(x, y, z, c, n_modes, eps=1e-6, iflag=1)
```

Computes: `f[k] = Σⱼ c[j] · exp(i · iflag · k · x[j])`

### Type 2: Uniform → Nonuniform 🔄

```python
# 1D
c = nufft1d2(x, f, eps=1e-6, iflag=1)

# 2D
c = nufft2d2(x, y, f, eps=1e-6, iflag=1)

# 3D
c = nufft3d2(x, y, z, f, eps=1e-6, iflag=1)
```

Computes: `c[j] = Σₖ f[k] · exp(i · iflag · k · x[j])`

### Parameters 🎛️

| Parameter | Description |
|-----------|-------------|
| `x, y, z` | Nonuniform points in [0, 2π) |
| `c` | Complex strengths at nonuniform points |
| `f` | Fourier mode coefficients |
| `n_modes` | Number of output modes (int or tuple) |
| `eps` | Requested precision (1e-2 to 1e-14) |
| `iflag` | Sign of exponent: +1 or -1 |

## 🔧 How It Works

nufftax implements the NUFFT using the standard approach:

1. 📡 **Spreading/Interpolation** - Convolve with a compactly-supported kernel (exponential of semicircle)
2. 🔢 **FFT** - Standard FFT on the oversampled grid
3. ➗ **Deconvolution** - Divide by kernel Fourier coefficients

The magic is that everything is written in pure JAX, so autodiff propagates through naturally. ✨

## ⚡ Performance

It's fast enough for most research purposes. For production workloads with massive transforms, you might want [finufft](https://github.com/flatironinstitute/finufft) - but then you lose the autodiff superpowers. 🦸

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=nufftax
```

## 📄 License

MIT

## 🙏 Acknowledgments

Algorithm based on the excellent [FINUFFT](https://github.com/flatironinstitute/finufft) library by the Flatiron Institute. 💙
