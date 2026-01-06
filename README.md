<p align="center">
  <img src="docs/_static/logo.png" alt="nufftax logo" width="200">
</p>

<h1 align="center">nufftax</h1>

<p align="center">
  <strong>Pure JAX implementation of the Non-Uniform Fast Fourier Transform (NUFFT)</strong>
</p>

<p align="center">
  <a href="https://github.com/geoffroyO/nufftax/actions/workflows/ci.yml"><img src="https://github.com/geoffroyO/nufftax/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://nufftax.readthedocs.io"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Documentation"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

<p align="center">
  <img src="docs/_static/mri_example.png" alt="MRI reconstruction example" width="100%">
</p>

## Why nufftax?

A JAX package for NUFFT already exists: [jax-finufft](https://github.com/flatironinstitute/jax-finufft). However, it wraps the C++ FINUFFT library via Foreign Function Interface (FFI), exposing it through custom XLA calls. This approach can lead to:
- **Kernel fusion issues on GPU** — custom XLA calls act as optimization barriers, preventing XLA from fusing operations
- **CUDA version matching** — GPU support requires matching CUDA versions between JAX and the library

**nufftax** takes a different approach — pure JAX implementation:

- **Fully differentiable** — gradients w.r.t. both values *and* sample locations
- **Pure JAX** — works with `jit`, `grad`, `vmap`, `jvp`, `vjp` with no FFI barriers
- **GPU ready** — runs on CPU/GPU without code changes, benefits from XLA fusion
- **All NUFFT types** — Type 1, 2, 3 in 1D, 2D, 3D

## JAX Transformation Support

| Transform | `jit` | `grad`/`vjp` | `jvp` | `vmap` |
|-----------|:-----:|:------------:|:-----:|:------:|
| **Type 1** (1D/2D/3D) | ✅ | ✅ | ✅ | ✅ |
| **Type 2** (1D/2D/3D) | ✅ | ✅ | ✅ | ✅ |
| **Type 3** (1D/2D/3D) | ✅ | ✅ | ✅ | ✅ |

**Differentiable inputs:**
- Type 1: `grad` w.r.t. `c` (strengths) and `x, y, z` (coordinates)
- Type 2: `grad` w.r.t. `f` (Fourier modes) and `x, y, z` (coordinates)
- Type 3: `grad` w.r.t. `c` (strengths), `x, y, z` (source coordinates), and `s, t, u` (target frequencies)

## Installation

```bash
uv pip install nufftax
```

## Quick Example

```python
import jax
import jax.numpy as jnp
from nufftax import nufft1d1

# Irregular sample locations in [-pi, pi)
x = jnp.array([0.1, 0.7, 1.3, 2.1, -0.5])
c = jnp.array([1.0+0.5j, 0.3-0.2j, 0.8+0.1j, 0.2+0.4j, 0.5-0.3j])

# Compute Fourier modes
f = nufft1d1(x, c, n_modes=32, eps=1e-6)

# Differentiate through the transform
grad_c = jax.grad(lambda c: jnp.sum(jnp.abs(nufft1d1(x, c, n_modes=32)) ** 2))(c)
```

## Documentation

**[Read the full documentation →](https://nufftax.readthedocs.io)**

- [Quickstart](https://nufftax.readthedocs.io/en/latest/quickstart.html) — get running in 5 minutes
- [Concepts](https://nufftax.readthedocs.io/en/latest/concepts.html) — understand the mathematics
- [Tutorials](https://nufftax.readthedocs.io/en/latest/tutorials.html) — MRI reconstruction, spectral analysis, optimization
- [API Reference](https://nufftax.readthedocs.io/en/latest/api.html) — complete function reference

## License

MIT. Algorithm based on [FINUFFT](https://github.com/flatironinstitute/finufft) by the Flatiron Institute.

## Citation

If you use nufftax in your research, please cite:

```bibtex
@software{nufftax,
  author = {Oudoumanessah, Geoffroy},
  title = {nufftax: Pure JAX implementation of the Non-Uniform Fast Fourier Transform},
  url = {https://github.com/geoffroyO/nufftax},
  year = {2025}
}
```
