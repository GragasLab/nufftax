"""Core NUFFT components: kernel, spreading, deconvolution."""

from .deconvolve import (
    deconvolve_pad_1d,
    deconvolve_pad_2d,
    deconvolve_pad_3d,
    deconvolve_shuffle_1d,
    deconvolve_shuffle_2d,
    deconvolve_shuffle_3d,
)
from .kernel import (
    KernelParams,
    compute_kernel_params,
    es_kernel,
    es_kernel_with_derivative,
    kernel_fourier_series,
)
from .spread import (
    fold_rescale,
    interp_1d,
    interp_2d,
    interp_3d,
    spread_1d,
    spread_2d,
    spread_3d,
)

__all__ = [
    # Kernel functions
    "es_kernel",
    "es_kernel_with_derivative",
    "compute_kernel_params",
    "kernel_fourier_series",
    "KernelParams",
    # Spreading functions
    "spread_1d",
    "spread_2d",
    "spread_3d",
    # Interpolation functions
    "interp_1d",
    "interp_2d",
    "interp_3d",
    "fold_rescale",
    # Deconvolution functions
    "deconvolve_shuffle_1d",
    "deconvolve_pad_1d",
    "deconvolve_shuffle_2d",
    "deconvolve_pad_2d",
    "deconvolve_shuffle_3d",
    "deconvolve_pad_3d",
]
