"""Grid size utilities for NUFFT."""

import math


def next_smooth_int(n: int) -> int:
    """
    Round n up to the next integer that is a product of small primes (2, 3, 5).

    These "smooth" numbers are efficient for FFT computation.

    Args:
        n: Input integer

    Returns:
        Smallest integer >= n that factors only into 2, 3, 5
    """
    if n <= 1:
        return 1

    # Check if n is already smooth
    def is_smooth(x):
        for p in [2, 3, 5]:
            while x % p == 0:
                x //= p
        return x == 1

    # Find next smooth number
    candidate = n
    while not is_smooth(candidate):
        candidate += 1

    return candidate


def compute_grid_size(
    n_modes: int,
    upsampfac: float,
    nspread: int,
) -> int:
    """
    Compute fine grid size for NUFFT.

    The grid must be:
    1. At least upsampfac * n_modes (for sufficient oversampling)
    2. At least 2 * nspread (for kernel support)
    3. A product of small primes for FFT efficiency

    Args:
        n_modes: Number of Fourier modes requested
        upsampfac: Upsampling factor (typically 2.0)
        nspread: Kernel width in grid points

    Returns:
        Fine grid size nf
    """
    nf = max(int(math.ceil(upsampfac * n_modes)), 2 * nspread)
    return next_smooth_int(nf)
