"""Public NUFFT API.

Thin wrappers around the JAX core primitives defined in `primitives.py`.
The primitives carry explicit JVP and transpose rules, so all of `jit`,
`grad`, `vjp`, `jvp`, `linear_transpose`, and `lax.custom_linear_solve`
(hence `jax.scipy.sparse.linalg.cg`/`gmres`/`bicgstab`) work through them.

Type 1 (nufftXd1) and Type 2 (nufftXd2) are mutual adjoints; Type 3 is
self-adjoint with source/target points swapped.

The `nufftXdY_jvp` aliases exist for backward compatibility with code that
imported the previous `custom_jvp`-only variants. They are now identical to
their non-`_jvp` counterparts.
"""

from jax import Array

from . import primitives as P


# ============================================================================
# Type 1 (nonuniform -> uniform)
# ============================================================================


def nufft1d1(x: Array, c: Array, n_modes: int, eps: float = 1e-6, isign: int = 1) -> Array:
    return P.nufft1d1_p.bind(x, c, n_modes=n_modes, eps=eps, isign=isign)


def nufft2d1(x: Array, y: Array, c: Array, n_modes: tuple[int, int], eps: float = 1e-6, isign: int = 1) -> Array:
    return P.nufft2d1_p.bind(x, y, c, n_modes=tuple(n_modes), eps=eps, isign=isign)


def nufft3d1(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
) -> Array:
    return P.nufft3d1_p.bind(x, y, z, c, n_modes=tuple(n_modes), eps=eps, isign=isign)


# ============================================================================
# Type 2 (uniform -> nonuniform)
# ============================================================================


def nufft1d2(x: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    return P.nufft1d2_p.bind(x, f, eps=eps, isign=isign)


def nufft2d2(x: Array, y: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    return P.nufft2d2_p.bind(x, y, f, eps=eps, isign=isign)


def nufft3d2(x: Array, y: Array, z: Array, f: Array, eps: float = 1e-6, isign: int = -1) -> Array:
    return P.nufft3d2_p.bind(x, y, z, f, eps=eps, isign=isign)


# ============================================================================
# Type 3 (nonuniform -> nonuniform)
# ============================================================================


def nufft1d3(
    x: Array,
    c: Array,
    s: Array,
    n_modes: int,
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    return P.nufft1d3_p.bind(x, c, s, n_modes=int(n_modes), eps=eps, isign=isign, upsampfac=upsampfac)


def nufft2d3(
    x: Array,
    y: Array,
    c: Array,
    s: Array,
    t: Array,
    n_modes: tuple[int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    return P.nufft2d3_p.bind(x, y, c, s, t, n_modes=tuple(n_modes), eps=eps, isign=isign, upsampfac=upsampfac)


def nufft3d3(
    x: Array,
    y: Array,
    z: Array,
    c: Array,
    s: Array,
    t: Array,
    u: Array,
    n_modes: tuple[int, int, int],
    eps: float = 1e-6,
    isign: int = 1,
    upsampfac: float = 2.0,
) -> Array:
    return P.nufft3d3_p.bind(x, y, z, c, s, t, u, n_modes=tuple(n_modes), eps=eps, isign=isign, upsampfac=upsampfac)


# Backward-compatibility aliases. Identical to the non-_jvp variants now that
# both forward and reverse mode AD flow through the same primitive.
nufft1d1_jvp = nufft1d1
nufft1d2_jvp = nufft1d2
nufft2d1_jvp = nufft2d1
nufft2d2_jvp = nufft2d2
nufft3d1_jvp = nufft3d1
nufft3d2_jvp = nufft3d2
nufft1d3_jvp = nufft1d3
nufft2d3_jvp = nufft2d3
nufft3d3_jvp = nufft3d3


__all__ = [
    "nufft1d1",
    "nufft1d2",
    "nufft1d3",
    "nufft2d1",
    "nufft2d2",
    "nufft2d3",
    "nufft3d1",
    "nufft3d2",
    "nufft3d3",
    "nufft1d1_jvp",
    "nufft1d2_jvp",
    "nufft1d3_jvp",
    "nufft2d1_jvp",
    "nufft2d2_jvp",
    "nufft2d3_jvp",
    "nufft3d1_jvp",
    "nufft3d2_jvp",
    "nufft3d3_jvp",
]
