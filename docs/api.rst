API Reference
=============

Complete reference for all public functions in nufftax.

Overview
--------

nufftax provides three types of Non-Uniform FFT:

.. list-table::
   :widths: 15 35 50
   :header-rows: 1

   * - Type
     - Direction
     - Functions
   * - Type 1
     - Nonuniform → Uniform
     - :func:`nufft1d1`, :func:`nufft2d1`, :func:`nufft3d1`
   * - Type 2
     - Uniform → Nonuniform
     - :func:`nufft1d2`, :func:`nufft2d2`, :func:`nufft3d2`
   * - Type 3
     - Nonuniform → Nonuniform
     - :func:`nufft1d3`, :func:`nufft2d3`, :func:`nufft3d3`

JAX Transformation Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

All functions support JAX transformations. Use ``jax.grad``, ``jax.vjp``, or ``jax.jvp`` directly:

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

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Type
     - Differentiable w.r.t.
   * - Type 1
     - ``c`` (strengths), ``x``, ``y``, ``z`` (coordinates)
   * - Type 2
     - ``f`` (Fourier modes), ``x``, ``y``, ``z`` (coordinates)
   * - Type 3
     - ``c`` (strengths), ``x``, ``y``, ``z`` (source coordinates), ``s``, ``t``, ``u`` (target frequencies)

**Example:**

.. code-block:: python

   import jax
   from nufftax import nufft1d1

   x = jnp.array([0.1, 0.5, 1.0, 2.0])
   c = jnp.array([1+1j, 2-1j, 0.5, 1j])

   # Gradient w.r.t. strengths
   grad_c = jax.grad(lambda c: jnp.sum(jnp.abs(nufft1d1(x, c, 32))**2))(c)

   # Gradient w.r.t. coordinates
   grad_x = jax.grad(lambda x: jnp.sum(jnp.abs(nufft1d1(x, c, 32))**2))(x)

   # Forward-mode AD (JVP)
   primals, tangents = jax.jvp(
       lambda c: nufft1d1(x, c, 32),
       (c,), (jnp.ones_like(c),)
   )

   # Reverse-mode AD (VJP)
   primals, vjp_fn = jax.vjp(lambda c: nufft1d1(x, c, 32), c)
   (grad_c,) = vjp_fn(jnp.ones_like(primals))

----

Type 1: Nonuniform → Uniform
----------------------------

Compute Fourier coefficients from scattered data.

**Mathematical definition:**

.. math::

   f_k = \sum_{j=1}^{M} c_j \cdot e^{i \cdot \text{isign} \cdot k \cdot x_j}

where :math:`k = -N/2, \ldots, N/2-1`.

.. currentmodule:: nufftax

1D Transform
~~~~~~~~~~~~

.. autofunction:: nufft1d1

**Example:**

.. code-block:: python

   from nufftax import nufft1d1
   import jax.numpy as jnp

   x = jnp.array([0.1, 0.5, 1.0, 2.0, -1.5])
   c = jnp.array([1+1j, 2-1j, 0.5, 1j, -1+0.5j])

   f = nufft1d1(x, c, n_modes=64, eps=1e-6, isign=1)
   # f.shape = (64,)

2D Transform
~~~~~~~~~~~~

.. autofunction:: nufft2d1

**Example:**

.. code-block:: python

   from nufftax import nufft2d1

   x = jnp.array([0.1, 0.5, 1.0])
   y = jnp.array([0.2, -0.3, 0.8])
   c = jnp.array([1+1j, 2-1j, 0.5])

   f = nufft2d1(x, y, c, n_modes=(32, 32), eps=1e-6)
   # f.shape = (32, 32)

3D Transform
~~~~~~~~~~~~

.. autofunction:: nufft3d1

----

Type 2: Uniform → Nonuniform
----------------------------

Evaluate Fourier series at scattered points.

**Mathematical definition:**

.. math::

   c_j = \sum_{k} f_k \cdot e^{i \cdot \text{isign} \cdot k \cdot x_j}

1D Transform
~~~~~~~~~~~~

.. autofunction:: nufft1d2

**Example:**

.. code-block:: python

   from nufftax import nufft1d2

   x = jnp.array([0.1, 0.5, 1.0, 2.0, -1.5])
   f = jnp.ones(64, dtype=jnp.complex64)

   c = nufft1d2(x, f, eps=1e-6, isign=-1)
   # c.shape = (5,)

2D Transform
~~~~~~~~~~~~

.. autofunction:: nufft2d2

3D Transform
~~~~~~~~~~~~

.. autofunction:: nufft3d2

----

Type 3: Nonuniform → Nonuniform
-------------------------------

Transform between two sets of nonuniform points.

**Mathematical definition:**

.. math::

   f_k = \sum_{j=1}^{M} c_j \cdot e^{i \cdot \text{isign} \cdot s_k \cdot x_j}

.. important::

   Type 3 transforms require pre-computing the internal grid size for JIT compilation.
   Use the helper functions below.

1D Transform
~~~~~~~~~~~~

.. autofunction:: nufft1d3

**Example:**

.. code-block:: python

   from nufftax import nufft1d3, compute_type3_grid_size

   x = jnp.array([0.1, 0.5, 1.0, 2.0])
   c = jnp.array([1+1j, 2-1j, 0.5, 1j])
   s = jnp.linspace(-10, 10, 50)

   # Pre-compute grid size
   n_modes = compute_type3_grid_size(x, s, eps=1e-6)

   f = nufft1d3(x, c, s, n_modes=n_modes, eps=1e-6)
   # f.shape = (50,)

2D Transform
~~~~~~~~~~~~

.. autofunction:: nufft2d3

3D Transform
~~~~~~~~~~~~

.. autofunction:: nufft3d3

----

Utility Functions
-----------------

Grid Size Computation
~~~~~~~~~~~~~~~~~~~~~

These functions compute the internal grid size needed for Type 3 transforms.
Call them before JIT-compiling your code.

.. autofunction:: compute_type3_grid_size

.. autofunction:: compute_type3_grid_sizes_2d

.. autofunction:: compute_type3_grid_sizes_3d

----

Common Parameters
-----------------

All transform functions share these parameters:

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``x``, ``y``, ``z``
     - ``Array``
     - Nonuniform point coordinates in :math:`[-\pi, \pi)`. Shape ``(M,)`` where M is the number of points.
   * - ``c``
     - ``Array``
     - Complex values at nonuniform points. Shape ``(M,)`` or ``(n_trans, M)`` for batched transforms.
   * - ``f``
     - ``Array``
     - Fourier mode coefficients on uniform grid.
   * - ``s``, ``t``, ``u``
     - ``Array``
     - Target frequencies for Type 3 transforms.
   * - ``n_modes``
     - ``int`` or ``tuple``
     - Number of output Fourier modes. For 2D/3D, use tuple like ``(Nx, Ny)`` or ``(Nx, Ny, Nz)``.
   * - ``eps``
     - ``float``
     - Requested relative precision. Range: ``1e-14`` to ``1e-1``. Default: ``1e-6``.
   * - ``isign``
     - ``int``
     - Sign of the exponent: ``+1`` or ``-1``. Default varies by transform type.

Return Values
~~~~~~~~~~~~~

- **Type 1:** Returns complex array of Fourier modes with shape ``(n_modes,)`` for 1D, ``(n_modes_x, n_modes_y)`` for 2D, etc.
- **Type 2:** Returns complex array of values at query points with shape ``(M,)``.
- **Type 3:** Returns complex array at target frequencies with shape ``(n_targets,)``.
