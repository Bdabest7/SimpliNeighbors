# SimpliNeighbors — Numba-accelerated bilateral filter
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Provides two backends — selected automatically at import time:
#
#   CUDA  (RTX 5090 / any NVIDIA GPU with numba + CUDA toolkit)
#       Processes the entire raster in one GPU dispatch.
#       One CUDA thread per output pixel, spatial kernel cached in registers.
#       Expected: ~3–5 s on RTX 5090 for a 22k×17k golf-course DTM.
#
#   Numba CPU  (any machine with numba installed, no GPU required)
#       JIT-compiled inner loop with fastmath (SVML fast-exp, FMA).
#       Works on individual tiles dispatched by ThreadPoolExecutor.
#       Expected: ~5–8× faster than the pure-NumPy fallback.
#
# Install:
#   pip install numba            # CPU acceleration (any machine)
#   # For CUDA also install CUDA Toolkit 12.8+ from developer.nvidia.com

from __future__ import annotations
import math
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# CUDA toolkit discovery — MUST happen before importing numba
#
# QGIS 4.x on Windows overrides the system PATH, so numba can't find the
# CUDA runtime / NVVM compiler DLLs.  PATH injection alone is not enough —
# numba locates NVVM via its own search (ctypes.util.find_library, etc.).
#
# The reliable fix is to set these env vars BEFORE numba is imported:
#   NUMBA_CUDA_DRIVER  — path to nvcuda.dll  (usually already loadable)
#   CUDA_HOME          — root of CUDA toolkit (numba derives NVVM path)
#
# We also inject into PATH and call os.add_dll_directory() as belt+suspenders.
# ---------------------------------------------------------------------------

def _find_cuda_root() -> str:
    """Return the CUDA toolkit root directory, or '' if not found."""
    # 1. CUDA_PATH env var (set by CUDA installer, QGIS usually preserves it)
    for var in ("CUDA_PATH", "CUDA_HOME"):
        p = os.environ.get(var, "")
        if p and os.path.isdir(p):
            return p

    # 2. Scan common install location
    prog = os.environ.get("ProgramFiles", r"C:\Program Files")
    toolkit = os.path.join(prog, "NVIDIA GPU Computing Toolkit", "CUDA")
    if os.path.isdir(toolkit):
        versions = sorted(
            (d for d in os.listdir(toolkit) if os.path.isdir(os.path.join(toolkit, d))),
            reverse=True,
        )
        if versions:
            return os.path.join(toolkit, versions[0])
    return ""


def _find_nvvm_dll(cuda_root: str) -> str:
    """Find the NVVM shared library inside the CUDA toolkit.

    CUDA 13.2 layout:  nvvm/bin/x64/nvvm64_40_0.dll
    Older layouts:     nvvm/bin/nvvm64_40_0.dll  or  nvvm/bin/nvvm.dll
    Searches all subdirectories under nvvm/bin.
    """
    nvvm_bin = os.path.join(cuda_root, "nvvm", "bin")
    if not os.path.isdir(nvvm_bin):
        return ""
    candidates = []
    for dirpath, _dirs, files in os.walk(nvvm_bin):
        for f in files:
            fl = f.lower()
            if fl.endswith(".dll") and fl.startswith("nvvm"):
                candidates.append(os.path.join(dirpath, f))
    # Prefer nvvm64_*.dll (versioned), fall back to nvvm.dll (unversioned)
    candidates.sort(key=lambda p: (0 if "64" in os.path.basename(p).lower() else 1, p))
    return candidates[0] if candidates else ""


if sys.platform == "win32":
    _cuda_root = _find_cuda_root()

    if _cuda_root:
        # --- Set CUDA_HOME so numba can derive NVVM + libdevice paths ---
        os.environ["CUDA_HOME"] = _cuda_root

        # --- Force compute capability for forward-compat with new GPUs ---
        # Numba's COMPUTE_CAPABILITIES table may not include the current GPU
        # (e.g. RTX 5090 = Blackwell sm_100).  NVVM generates forward-compatible
        # PTX, so targeting the highest CC numba knows (sm_90) works fine.
        if not os.environ.get("NUMBA_FORCE_CUDA_CC"):
            os.environ["NUMBA_FORCE_CUDA_CC"] = "9.0"

        # --- Store NVVM DLL info for post-import patching (see below) ---
        _nvvm_dll = _find_nvvm_dll(_cuda_root)

        # --- Inject DLL directories into PATH and os.add_dll_directory ---
        _current_path = os.environ.get("PATH", "")
        _dirs_to_add = []
        for _subdir in (
            "bin",
            os.path.join("nvvm", "bin"),
            os.path.join("nvvm", "bin", "x64"),
            os.path.join("lib", "x64"),
        ):
            _dll_dir = os.path.join(_cuda_root, _subdir)
            if os.path.isdir(_dll_dir) and _dll_dir.lower() not in _current_path.lower():
                _dirs_to_add.append(_dll_dir)
                try:
                    os.add_dll_directory(_dll_dir)
                except (OSError, AttributeError):
                    pass
        if _dirs_to_add:
            os.environ["PATH"] = ";".join(_dirs_to_add) + ";" + _current_path

# ---------------------------------------------------------------------------
# Availability detection (done once at import time)
# ---------------------------------------------------------------------------

_NUMBA_OK = False
_CUDA_OK  = False

try:
    import numba
    _NUMBA_OK = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Monkey-patch numba's NVVM search path for CUDA 13.x layout
#
# CUDA 13.x puts nvvm64_40_0.dll in  nvvm/bin/x64/  but numba only searches
# nvvm/bin/ (flat listdir).  We patch _nvvm_lib_dir() to return the real
# directory components BEFORE anything calls get_cuda_paths() (which caches).
# ---------------------------------------------------------------------------

if _NUMBA_OK and sys.platform == "win32":
    _nvvm_dll_path = globals().get("_nvvm_dll", "") or _find_nvvm_dll(_find_cuda_root())

    if _nvvm_dll_path:
        try:
            import numba.cuda.cuda_paths as _cp
            # Clear any cached result so our patch takes effect
            if hasattr(_cp.get_cuda_paths, '_cached_result'):
                del _cp.get_cuda_paths._cached_result

            _nvvm_dll_dir = os.path.dirname(_nvvm_dll_path)
            _cuda_root_for_patch = os.environ.get("CUDA_HOME", "")
            if _cuda_root_for_patch:
                _rel = os.path.relpath(_nvvm_dll_dir, _cuda_root_for_patch)
                _parts = tuple(_rel.split(os.sep))

                def _patched_nvvm_lib_dir(_parts=_parts):
                    return _parts

                _cp._nvvm_lib_dir = _patched_nvvm_lib_dir
        except Exception:
            pass

if _NUMBA_OK:
    try:
        from numba import cuda as _cuda
        _CUDA_OK = _cuda.is_available()
    except Exception:
        _CUDA_OK = False

_TPB = 16   # CUDA thread-block side  (16×16 = 256 threads, safe on all archs)


# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------

if _CUDA_OK:
    from numba import cuda as _cuda

    @_cuda.jit(fastmath=True)
    def _cuda_kernel(data, padded, spatial_flat, inv_2_sc2, fs, h, w, out):
        """
        One CUDA thread per output pixel (i, j).

        data        : (h, w)  float32  — original raster
        padded      : (h+2R, w+2R)  float32  — reflect-padded raster
        spatial_flat: (fs*fs,)  float32  — pre-computed spatial Gaussian weights
        inv_2_sc2   : float32  — 1 / (2 * sigma_color²)
        fs          : int      — filter_size = 2*radius + 1
        out         : (h, w)  float32  — output (written by this kernel)
        """
        i, j = _cuda.grid(2)
        if i >= h or j >= w:
            return

        center = data[i, j]
        ov = numba.float32(0.0)
        wt = numba.float32(0.0)

        for dy in range(fs):
            for dx in range(fs):
                nb   = padded[i + dy, j + dx]
                diff = center - nb
                rw   = math.exp(-diff * diff * inv_2_sc2)
                tw   = spatial_flat[dy * fs + dx] * rw
                ov  += tw * nb
                wt  += tw

        out[i, j] = ov / wt


def bilateral_filter_cuda(
    data: np.ndarray,
    filter_size: int,
    sigma_space: float,
    sigma_color: float,
) -> np.ndarray:
    """
    Run bilateral filter on the GPU.

    Processes the *entire* raster in a single kernel launch — no CPU tiling.
    The full raster (~1.5 GB for a 22k×17k DTM) is transferred to VRAM once,
    filtered, then transferred back.  Net GPU time on RTX 5090: ~3–5 seconds.

    NOTE: First call JIT-compiles the PTX kernel (~10–30 s one-time cost).
    Subsequent calls hit the numba PTX cache and are instant.
    """
    from numba import cuda as _cuda_

    data   = data.astype(np.float32)
    h, w   = data.shape
    radius = filter_size // 2

    # Build padded array and spatial kernel on CPU before transfer
    padded = np.pad(data, radius, mode="reflect").astype(np.float32)

    ky, kx   = np.mgrid[0:filter_size, 0:filter_size]
    spatial  = np.exp(
        -((kx - radius) ** 2 + (ky - radius) ** 2) / (2.0 * float(sigma_space) ** 2)
    ).astype(np.float32).ravel()           # shape: (fs*fs,)

    inv_2_sc2 = np.float32(1.0 / (2.0 * float(sigma_color) ** 2))

    # Host → Device transfers
    d_data    = _cuda_.to_device(data)
    d_padded  = _cuda_.to_device(padded)
    d_spatial = _cuda_.to_device(spatial)
    d_out     = _cuda_.device_array((h, w), dtype=np.float32)

    # Launch: ceil(h/TPB) × ceil(w/TPB) blocks, each 16×16 threads
    grid  = (math.ceil(h / _TPB), math.ceil(w / _TPB))
    block = (_TPB, _TPB)

    _cuda_kernel[grid, block](
        d_data, d_padded, d_spatial, inv_2_sc2, filter_size, h, w, d_out
    )
    _cuda_.synchronize()

    return d_out.copy_to_host()


# ---------------------------------------------------------------------------
# Numba CPU backend
# ---------------------------------------------------------------------------

if _NUMBA_OK:
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def _cpu_kernel(data, padded, spatial_flat, inv_2_sc2, fs, h, w):
        """
        Numba JIT bilateral filter for a single tile.

        nopython  : no Python object overhead in the inner loop
        fastmath  : SVML approximate exp (~4× faster than libm), FMA
        cache     : compiled bytecode cached to disk — first call only is slow
        NO parallel/prange: ThreadPoolExecutor already saturates all cores
                            at the tile level; adding prange would oversubscribe
        """
        out = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                center = data[i, j]
                ov     = np.float32(0.0)
                wt     = np.float32(0.0)
                for dy in range(fs):
                    for dx in range(fs):
                        nb   = padded[i + dy, j + dx]
                        diff = center - nb
                        rw   = math.exp(-diff * diff * inv_2_sc2)
                        tw   = spatial_flat[dy * fs + dx] * rw
                        ov  += tw * nb
                        wt  += tw
                out[i, j] = ov / wt
        return out


def bilateral_filter_numba_cpu(
    data: np.ndarray,
    filter_size: int,
    sigma_space: float,
    sigma_color: float,
) -> np.ndarray:
    """Numba CPU bilateral filter — called per-tile by ThreadPoolExecutor."""
    data   = data.astype(np.float32)
    h, w   = data.shape
    radius = filter_size // 2

    padded = np.pad(data, radius, mode="reflect").astype(np.float32)

    ky, kx  = np.mgrid[0:filter_size, 0:filter_size]
    spatial = np.exp(
        -((kx - radius) ** 2 + (ky - radius) ** 2) / (2.0 * float(sigma_space) ** 2)
    ).astype(np.float32).ravel()

    inv_2_sc2 = np.float32(1.0 / (2.0 * float(sigma_color) ** 2))

    return _cpu_kernel(data, padded, spatial, inv_2_sc2, filter_size, h, w)
