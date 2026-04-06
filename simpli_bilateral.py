# SimpliNeighbors — bilateral filter dispatcher
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Call bilateral_filter() — it automatically selects the fastest available
# backend and caches the choice for the lifetime of the process:
#
#   "CUDA"      RTX / NVIDIA GPU via numba.cuda  (~250× vs NumPy baseline)
#   "Numba CPU" Numba JIT fastmath, per-tile      (~6–8× vs NumPy baseline)
#   "NumPy"     Pure NumPy fallback               (1× baseline, no extra deps)
#
# The CUDA path is handled specially in simpli_neighbors_algorithm.py:
# it processes the *entire* raster in one GPU dispatch.  The tile-level
# bilateral_filter() here is only called on the CPU paths.

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Backend detection — evaluated once, result cached in _BACKEND
# ---------------------------------------------------------------------------

_BACKEND: str | None = None   # "CUDA" | "Numba CPU" | "NumPy"


def get_bilateral_backend() -> str:
    """
    Return the name of the active bilateral backend.
    Cached after the first call — safe to call repeatedly.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    # 1. CUDA
    try:
        from .simpli_bilateral_numba import _CUDA_OK
        if _CUDA_OK:
            # Confirm the public function imports cleanly
            from .simpli_bilateral_numba import bilateral_filter_cuda  # noqa: F401
            _BACKEND = "CUDA"
            return _BACKEND
    except Exception:
        pass

    # 2. Numba CPU
    try:
        from .simpli_bilateral_numba import _NUMBA_OK
        if _NUMBA_OK:
            from .simpli_bilateral_numba import bilateral_filter_numba_cpu  # noqa: F401
            _BACKEND = "Numba CPU"
            return _BACKEND
    except Exception:
        pass

    # 3. NumPy fallback
    _BACKEND = "NumPy"
    return _BACKEND


# ---------------------------------------------------------------------------
# Public API — tile-level (CPU paths only; CUDA bypasses this via algorithm)
# ---------------------------------------------------------------------------

def bilateral_filter(
    data: np.ndarray,
    filter_size: int,
    sigma_space: float,
    sigma_color: float,
) -> np.ndarray:
    """
    Apply bilateral filter to a single tile (CPU paths).

    The CUDA path never calls this function — it processes the full raster
    directly via simpli_bilateral_numba.bilateral_filter_cuda().
    """
    backend = get_bilateral_backend()

    if backend == "Numba CPU":
        from .simpli_bilateral_numba import bilateral_filter_numba_cpu
        return bilateral_filter_numba_cpu(data, filter_size, sigma_space, sigma_color)

    return _bilateral_numpy(data, filter_size, sigma_space, sigma_color)


# ---------------------------------------------------------------------------
# NumPy implementation (pure, no extra dependencies)
# ---------------------------------------------------------------------------

def _bilateral_numpy(
    data: np.ndarray,
    filter_size: int,
    sigma_space: float,
    sigma_color: float,
) -> np.ndarray:
    """
    Pure NumPy bilateral filter — the guaranteed fallback.

    Algorithm
    ---------
    Iterates over each (dy, dx) offset in the filter window.
    Each iteration is one fully-vectorised NumPy operation over the tile —
    no Python-level pixel loop.

    Weights:
      w_spatial(p,q) = exp( -||p-q||² / 2σ_s² )   spatial Gaussian
      w_range(p,q)   = exp( -(I(p)-I(q))² / 2σ_c² ) range  Gaussian
    """
    data   = data.astype(np.float32)
    radius = filter_size // 2
    h, w   = data.shape

    padded = np.pad(data, radius, mode="reflect")

    ky, kx = np.mgrid[0:filter_size, 0:filter_size]
    spatial_kernel = np.exp(
        -((kx - radius) ** 2 + (ky - radius) ** 2) / (2.0 * float(sigma_space) ** 2)
    ).astype(np.float32)

    inv_2_sc2 = np.float32(1.0 / (2.0 * float(sigma_color) ** 2))

    output = np.zeros((h, w), dtype=np.float32)
    wsum   = np.zeros((h, w), dtype=np.float32)

    for dy in range(filter_size):
        for dx in range(filter_size):
            neighbour = padded[dy: dy + h, dx: dx + w]
            diff      = data - neighbour
            r_w       = np.exp(-(diff * diff) * inv_2_sc2)
            total_w   = np.float32(spatial_kernel[dy, dx]) * r_w
            output   += total_w * neighbour
            wsum     += total_w

    return output / wsum
