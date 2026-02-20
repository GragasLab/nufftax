nufftax
=======

**A pure JAX implementation of the Non-Uniform Fast Fourier Transform.**

.. image:: https://github.com/GragasLab/nufftax/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/GragasLab/nufftax/actions/workflows/ci.yml

.. image:: https://img.shields.io/badge/docs-online-blue.svg
   :target: https://nufftax.readthedocs.io

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

----

What is nufftax?
----------------

**nufftax** lets you compute Fourier transforms on data sampled at *arbitrary* (non-uniform) locations,
while keeping full compatibility with JAX's automatic differentiation and GPU acceleration.

Traditional FFTs require data on a regular grid. But real-world data is often irregularly sampled:

- **MRI scans** acquire k-space data along spiral or radial trajectories
- **Astronomical observations** happen at irregular time intervals
- **Sensor networks** collect measurements at scattered spatial locations

The NUFFT bridges this gap, and **nufftax** makes it differentiable.

Why nufftax?
------------

A JAX package for NUFFT already exists: `jax-finufft <https://github.com/flatironinstitute/jax-finufft>`_.
However, it wraps the C++ FINUFFT library via Foreign Function Interface (FFI), exposing it through custom XLA calls. This approach can lead to:

- **Kernel fusion issues on GPU** — custom XLA calls act as optimization barriers, preventing XLA from fusing operations
- **CUDA version matching** — GPU support requires matching CUDA versions between JAX and the library

**nufftax** takes a different approach — pure JAX implementation:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Feature
     - Benefit
   * - **Fully differentiable**
     - Compute gradients through the entire transform - both w.r.t. data values *and* sampling locations
   * - **JAX native**
     - Works with ``jit``, ``grad``, ``vmap``, ``jvp``, ``vjp`` with no FFI barriers
   * - **GPU ready**
     - Runs on CPU/GPU without code changes, benefits from XLA fusion
   * - **Pallas GPU kernels**
     - Fused Triton spreading kernels with 5–75× speedups on A100/H100
   * - **No compilation step**
     - Pure Python/JAX - no C++ extensions to build

JAX Transformation Support
--------------------------

.. list-table::
   :widths: 30 15 15 15 15
   :header-rows: 1

   * - Transform
     - ``jit``
     - ``grad``/``vjp``
     - ``jvp``
     - ``vmap``
   * - **Type 1** (1D/2D/3D)
     - ✅
     - ✅
     - ✅
     - ✅
   * - **Type 2** (1D/2D/3D)
     - ✅
     - ✅
     - ✅
     - ✅
   * - **Type 3** (1D/2D/3D)
     - ✅
     - ✅
     - ✅
     - ✅

**Differentiable inputs:**

- **Type 1:** ``grad`` w.r.t. ``c`` (strengths) and ``x``, ``y``, ``z`` (coordinates)
- **Type 2:** ``grad`` w.r.t. ``f`` (Fourier modes) and ``x``, ``y``, ``z`` (coordinates)
- **Type 3:** ``grad`` w.r.t. ``c`` (strengths), ``x``, ``y``, ``z`` (source coordinates), and ``s``, ``t``, ``u`` (target frequencies)

Quick Example
-------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from nufftax import nufft1d1

   # Irregular sample locations in [-pi, pi)
   x = jnp.array([0.1, 0.7, 1.3, 2.1, -0.5])

   # Complex values at those locations
   c = jnp.array([1.0+0.5j, 0.3-0.2j, 0.8+0.1j, 0.2+0.4j, 0.5-0.3j])

   # Compute 32 Fourier modes
   f = nufft1d1(x, c, n_modes=32, eps=1e-6)

   # Differentiate through the transform
   def loss(c):
       return jnp.sum(jnp.abs(nufft1d1(x, c, n_modes=32)) ** 2)

   grad_c = jax.grad(loss)(c)

GPU Acceleration
----------------

On GPU, nufftax automatically dispatches spreading and interpolation to fused
`Pallas <https://docs.jax.dev/en/latest/pallas/index.html>`_ (Triton) kernels
when the problem is large enough. This avoids materializing
O(M × nspread\ :sup:`d`) intermediate tensors and uses atomic scatter-add for
spreading.

.. list-table::
   :widths: 30 20 30
   :header-rows: 1

   * - Operation
     - Backend
     - Speedup vs pure JAX
   * - 1D spread
     - A100
     - 5–67× (M ≥ 100K)
   * - 1D spread
     - H100
     - 4–75× (M ≥ 100K)
   * - 2D spread
     - A100/H100
     - 2–3× (M ≥ 100K)

The dispatch is transparent — no code changes required. On CPU or for small
problems, the pure JAX path is used.

Installation
------------

**CPU only:**

.. code-block:: bash

   uv pip install nufftax

**With CUDA 12 GPU support:**

.. code-block:: bash

   uv pip install "nufftax[cuda12]"

**Development install (from source):**

.. code-block:: bash

   git clone https://github.com/GragasLab/nufftax.git
   cd nufftax
   uv pip install -e ".[dev]"

This installs test dependencies (``pytest``, ``ruff``, ``finufft`` for
comparison testing, ``pre-commit``).

**Development install with CUDA 12:**

.. code-block:: bash

   uv pip install -e ".[dev,cuda12]"

**With docs dependencies:**

.. code-block:: bash

   uv pip install -e ".[docs]"

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   concepts
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

License
-------

MIT License. Algorithm based on `FINUFFT <https://github.com/flatironinstitute/finufft>`_ by the Flatiron Institute.
