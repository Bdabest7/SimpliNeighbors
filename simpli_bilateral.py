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
import sys
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
# Pinning a backend (finding 8)
#
# Without a pin the backend is whatever this process finds, so results and speed
# can differ between an interactive session and a service account. A caller can
# pin one: the BACKEND parameter of the algorithm, or the environment variable
# SIMPLINEIGHBORS_BACKEND (auto | cuda | numba | numpy). A pinned backend that
# is not available is an error, never a silent fallback.
# ---------------------------------------------------------------------------

BACKEND_CUDA = "CUDA"
BACKEND_NUMBA = "Numba CPU"
BACKEND_NUMPY = "NumPy"

_ALIASES = {
    "cuda": BACKEND_CUDA, "gpu": BACKEND_CUDA,
    "numba": BACKEND_NUMBA, "numba cpu": BACKEND_NUMBA, "cpu": BACKEND_NUMBA,
    "numpy": BACKEND_NUMPY,
}


class BackendUnavailable(RuntimeError):
    """A pinned backend cannot be used on this machine / in this account."""


def normalize_backend_name(value) -> str | None:
    """'auto'/''/None -> None (no pin); otherwise one of the BACKEND_* names."""
    v = ("" if value is None else str(value)).strip().lower()
    if v in ("", "auto", "default"):
        return None
    if v not in _ALIASES:
        raise BackendUnavailable(
            f"Unknown backend {value!r}; use auto, cuda, numba or numpy")
    return _ALIASES[v]


def resolve_backend(requested=None) -> str:
    """Return the backend to use. requested=None means pick the best available.
    A pinned backend that is missing raises BackendUnavailable."""
    pin = normalize_backend_name(requested)
    if pin is None:
        return get_bilateral_backend()
    if pin == BACKEND_NUMPY:
        return pin
    try:
        from . import simpli_bilateral_numba as nb
    except Exception as e:
        raise BackendUnavailable(f"{pin} backend requested but numba support cannot be imported: {e}")
    if pin == BACKEND_NUMBA:
        if not nb._NUMBA_OK:
            raise BackendUnavailable("Numba CPU backend requested but numba is not installed in this Python")
        return pin
    if nb.CUDA_DISABLED:
        raise BackendUnavailable("CUDA backend requested but CUDA is switched off (SIMPLINEIGHBORS_CUDA)")
    if not nb._NUMBA_OK:
        raise BackendUnavailable("CUDA backend requested but numba is not installed in this Python")
    if not nb._CUDA_OK:
        why = "; ".join(nb.CUDA_NOTES) or "no usable CUDA toolkit/driver found"
        raise BackendUnavailable(f"CUDA backend requested but not available ({why})")
    return pin


def describe_backend(name: str) -> str:
    """One line for the log: what exactly will run."""
    parts = [f"python {sys.version.split()[0]}", f"numpy {np.__version__}"]
    try:
        from . import simpli_bilateral_numba as nb
        if nb._NUMBA_OK:
            import numba
            parts.append(f"numba {numba.__version__}")
            parts.append(f"numba cache {nb.NUMBA_CACHE_DIR}")
        if name == BACKEND_CUDA and nb._CUDA_OK:
            from numba import cuda
            dev = cuda.get_current_device()
            dname = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
            parts.append(f"GPU {dname}")
            parts.append(f"CUDA toolkit {nb.CUDA_ROOT or 'not set'}")
    except Exception as e:  # description only, never fatal
        parts.append(f"(details unavailable: {e})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Public API — tile-level (CPU paths only; CUDA bypasses this via algorithm)
# ---------------------------------------------------------------------------

def bilateral_filter(
    data: np.ndarray,
    filter_size: int,
    sigma_space: float,
    sigma_color: float,
    backend: str | None = None,
) -> np.ndarray:
    """
    Apply bilateral filter to a single tile (CPU paths).

    The CUDA path never calls this function — it processes the full raster
    directly via simpli_bilateral_numba.bilateral_filter_cuda().
    backend: a name from resolve_backend(), or None for the cached best one.
    """
    backend = backend or get_bilateral_backend()

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
