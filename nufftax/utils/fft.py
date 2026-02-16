"""Centered FFT/IFFT and complex ↔ real channel utilities.

Pure JAX implementations compatible with jit, grad, vmap.
"""

import jax.numpy as jnp
from jax import Array


def to_complex(x: Array) -> Array:
    """Convert 2-channel real/imag to complex. (..., 2) → (...)."""
    return x[..., 0] + 1j * x[..., 1]


def from_complex(x: Array) -> Array:
    """Convert complex to 2-channel real/imag. (...) → (..., 2)."""
    return jnp.stack([x.real, x.imag], axis=-1)


def fftc(x: Array, axes: tuple[int, ...] = (-2, -1)) -> Array:
    """Centered 2D FFT: image space → k-space."""
    return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)


def ifftc(x: Array, axes: tuple[int, ...] = (-2, -1)) -> Array:
    """Centered 2D IFFT: k-space → image space."""
    return jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
