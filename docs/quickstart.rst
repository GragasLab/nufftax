Quickstart
==========

This guide will get you up and running with nufftax in 5 minutes.

Your First Transform
--------------------

Let's compute the Fourier transform of data at irregular sample points.

.. code-block:: python

   import jax.numpy as jnp
   from nufftax import nufft1d1

   # Sample locations (not evenly spaced!)
   x = jnp.array([-2.1, -0.5, 0.3, 1.2, 2.8])

   # Complex values at each location
   c = jnp.array([1.0+0.0j, 0.5+0.5j, 2.0-1.0j, 0.3+0.2j, 1.5+0.0j])

   # Compute 32 Fourier modes
   f = nufft1d1(x, c, n_modes=32, eps=1e-6)

   print(f"Input: {len(c)} scattered points")
   print(f"Output: {len(f)} Fourier modes")

That's it! The result ``f`` contains the Fourier coefficients of your scattered data.

Going the Other Direction
-------------------------

What if you have Fourier modes and want to evaluate them at specific points?
That's Type 2:

.. code-block:: python

   from nufftax import nufft1d2

   # Start with some Fourier modes
   f = jnp.zeros(32, dtype=jnp.complex64)
   f = f.at[5].set(1.0)   # A single frequency component

   # Points where we want to evaluate
   x_eval = jnp.linspace(-jnp.pi, jnp.pi, 100)

   # Evaluate the Fourier series
   values = nufft1d2(x_eval, f, eps=1e-6)

   # 'values' contains the Fourier series evaluated at each point in x_eval

Computing Gradients
-------------------

One of nufftax's key features is automatic differentiation.

**Gradient w.r.t. values:**

.. code-block:: python

   import jax

   x = jnp.array([0.1, 0.5, 1.0, 2.0])
   c = jnp.array([1+1j, 2-1j, 0.5+0j, 1j])

   def loss(c):
       """Total power in Fourier domain."""
       f = nufft1d1(x, c, n_modes=32, eps=1e-6)
       return jnp.sum(jnp.abs(f) ** 2)

   # Compute gradient of loss w.r.t. c
   grad_c = jax.grad(loss)(c)

**Gradient w.r.t. sample locations:**

.. code-block:: python

   def loss_positions(x):
       """Loss as a function of sample positions."""
       f = nufft1d1(x, c, n_modes=32, eps=1e-6)
       return jnp.sum(jnp.abs(f) ** 2)

   # Compute gradient w.r.t. positions
   grad_x = jax.grad(loss_positions)(x)

This is powerful for optimization problems where you want to learn optimal sampling locations.

Batched Transforms
------------------

Process multiple transforms in parallel with ``vmap``:

.. code-block:: python

   # Multiple sets of sample locations
   x_batch = jnp.stack([
       jnp.array([0.1, 0.5, 1.0, 2.0]),
       jnp.array([0.2, 0.6, 1.1, 2.1]),
       jnp.array([0.3, 0.7, 1.2, 2.2]),
   ])  # Shape: (3, 4)

   c = jnp.array([1+1j, 2-1j, 0.5+0j, 1j])

   # Vectorize over the batch dimension
   batched_nufft = jax.vmap(lambda x: nufft1d1(x, c, n_modes=32))

   f_batch = batched_nufft(x_batch)  # Shape: (3, 32)

Multi-Dimensional Transforms
----------------------------

nufftax supports 2D and 3D:

.. code-block:: python

   from nufftax import nufft2d1, nufft3d1

   # 2D scattered points
   x = jnp.array([0.1, 0.5, 1.0])
   y = jnp.array([0.2, -0.3, 0.8])
   c = jnp.array([1+1j, 2-1j, 0.5+0j])

   f_2d = nufft2d1(x, y, c, n_modes=(16, 16), eps=1e-6)
   print(f"2D output shape: {f_2d.shape}")  # (16, 16)

   # 3D scattered points
   z = jnp.array([0.1, 0.4, -0.2])
   f_3d = nufft3d1(x, y, z, c, n_modes=(8, 8, 8), eps=1e-6)
   print(f"3D output shape: {f_3d.shape}")  # (8, 8, 8)

JIT Compilation
---------------

nufftax functions are compatible with JAX's JIT compilation. For best performance,
wrap your functions with ``@jax.jit``:

.. code-block:: python

   @jax.jit
   def my_analysis(x, c):
       f = nufft1d1(x, c, n_modes=64, eps=1e-6)
       return jnp.abs(f) ** 2  # Power spectrum

   # First call compiles, subsequent calls are fast
   power = my_analysis(x, c)

You can also JIT individual NUFFT calls:

.. code-block:: python

   # JIT a single function with static arguments
   jitted_nufft = jax.jit(nufft1d1, static_argnames=("n_modes", "eps", "isign"))
   f = jitted_nufft(x, c, n_modes=64, eps=1e-6)

Precision vs Speed
------------------

Control the accuracy/speed tradeoff with ``eps``:

.. code-block:: python

   # Fast, ~1% accuracy
   f_fast = nufft1d1(x, c, n_modes=64, eps=1e-2)

   # Balanced (default)
   f_balanced = nufft1d1(x, c, n_modes=64, eps=1e-6)

   # High precision, slower
   f_precise = nufft1d1(x, c, n_modes=64, eps=1e-12)

Next Steps
----------

- :doc:`concepts` - Understand the mathematics behind NUFFT
- :doc:`tutorials` - Practical examples for common applications
- :doc:`api` - Complete API reference
