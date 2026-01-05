"""NUFFT transform functions.

All transforms are JIT-compiled by default for 5-11x speedup.
"""

# Autodiff-enabled transforms (all JIT-compiled)
from .autodiff import (
    # Helper functions
    compute_position_gradient_1d,
    compute_position_gradient_2d,
    compute_position_gradient_3d,
    # 1D transforms with custom VJP
    nufft1d1,
    # 1D transforms with custom JVP
    nufft1d1_jvp,
    nufft1d2,
    nufft1d2_jvp,
    nufft1d3,
    nufft1d3_jvp,
    # 2D transforms with custom VJP
    nufft2d1,
    # 2D transforms with custom JVP
    nufft2d1_jvp,
    nufft2d2,
    nufft2d2_jvp,
    nufft2d3,
    nufft2d3_jvp,
    # 3D transforms with custom VJP
    nufft3d1,
    # 3D transforms with custom JVP
    nufft3d1_jvp,
    nufft3d2,
    nufft3d2_jvp,
    nufft3d3,
    nufft3d3_jvp,
)

__all__ = [
    # Type 1 transforms (nonuniform -> uniform)
    "nufft1d1",
    "nufft2d1",
    "nufft3d1",
    # Type 2 transforms (uniform -> nonuniform)
    "nufft1d2",
    "nufft2d2",
    "nufft3d2",
    # Type 3 transforms (nonuniform -> nonuniform)
    "nufft1d3",
    "nufft2d3",
    "nufft3d3",
    # JVP versions (1D)
    "nufft1d1_jvp",
    "nufft1d2_jvp",
    "nufft1d3_jvp",
    # JVP versions (2D)
    "nufft2d1_jvp",
    "nufft2d2_jvp",
    "nufft2d3_jvp",
    # JVP versions (3D)
    "nufft3d1_jvp",
    "nufft3d2_jvp",
    "nufft3d3_jvp",
    # Gradient helpers
    "compute_position_gradient_1d",
    "compute_position_gradient_2d",
    "compute_position_gradient_3d",
]
