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
import re
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Numba cache location — MUST happen before importing numba (finding 9)
#
# The CPU kernel is compiled with cache=True. Numba's on-disk cache is loaded
# with pickle, and by default it lives in __pycache__ inside the plugin folder:
# anyone who can write that folder could plant a cache that runs code in every
# account that loads the plugin. Each account therefore gets its own cache
# folder under its own profile (writable by that account only). An explicit
# NUMBA_CACHE_DIR (the automation sets one per account) is respected.
# ---------------------------------------------------------------------------

def _default_numba_cache_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "SimpliNeighbors", "numba-cache")


def _set_numba_cache_dir() -> str:
    path = os.environ.get("NUMBA_CACHE_DIR") or _default_numba_cache_dir()
    path = os.path.abspath(path)
    os.environ["NUMBA_CACHE_DIR"] = path
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass  # numba falls back to its own choice (and then caches nothing) if unusable
    # If something else already imported numba, it read the env var too late.
    if "numba" in sys.modules:
        try:
            sys.modules["numba"].config.CACHE_DIR = path
        except Exception:
            pass
    return path


NUMBA_CACHE_DIR = _set_numba_cache_dir()

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

def _cuda_disabled() -> bool:
    """SIMPLINEIGHBORS_CUDA=0 (or off/false/no) turns the CUDA path off entirely:
    no toolkit discovery, no PATH changes, no CUDA probing (finding 12)."""
    return os.environ.get("SIMPLINEIGHBORS_CUDA", "").strip().lower() in ("0", "off", "false", "no")


def _program_files_roots() -> list:
    r"""Real paths of the folders a CUDA toolkit may be loaded from.

    The fixed C:\Program Files, plus ProgramW6432 / ProgramFiles when they
    really are a folder called 'Program Files' directly under a drive root, so
    a tampered environment variable cannot add an arbitrary folder.
    """
    cands = [r"C:\Program Files"]
    for var in ("ProgramW6432", "ProgramFiles"):
        p = os.environ.get(var, "")
        if not p:
            continue
        p = os.path.normpath(p)
        drive, tail = os.path.splitdrive(p)
        if os.path.basename(p).lower() == "program files" and os.path.dirname(p) == drive + os.sep:
            cands.append(p)
    roots = []
    for c in cands:
        if os.path.isdir(c):
            r = os.path.normcase(os.path.realpath(c))
            if r not in roots:
                roots.append(r)
    return roots


def _is_trusted_cuda_dir(path: str) -> bool:
    """True if path (links and junctions resolved) is inside Program Files."""
    try:
        real = os.path.normcase(os.path.realpath(path))
        for root in _program_files_roots():
            if os.path.commonpath([real, root]) == root:
                return True
    except (ValueError, OSError):
        pass
    return False


def _version_key(name: str):
    """'v13.2' -> (13, 2), so v9.0 sorts below v12.8 and v13.2 (not as strings)."""
    nums = re.findall(r"\d+", name)
    return tuple(int(n) for n in nums)


CUDA_NOTES: list = []   # what discovery decided, for the algorithm's log


def _find_cuda_root() -> str:
    """Return the CUDA toolkit root directory, or '' if not found.

    Only folders under Program Files are accepted, from CUDA_PATH / CUDA_HOME
    or from the standard install location (newest version by number).
    """
    # 1. CUDA_PATH env var (set by CUDA installer, QGIS usually preserves it)
    for var in ("CUDA_PATH", "CUDA_HOME"):
        p = os.environ.get(var, "")
        if not p:
            continue
        if not os.path.isdir(p):
            CUDA_NOTES.append(f"{var}={p} ignored: not a folder")
        elif not _is_trusted_cuda_dir(p):
            CUDA_NOTES.append(f"{var}={p} ignored: not under Program Files")
        elif not os.path.isdir(os.path.join(p, "nvvm")):
            CUDA_NOTES.append(f"{var}={p} ignored: no nvvm folder, not a CUDA toolkit")
        else:
            return p

    # 2. Scan the standard install location, newest version first (numeric order)
    for root in _program_files_roots():
        toolkit = os.path.join(root, "NVIDIA GPU Computing Toolkit", "CUDA")
        if not os.path.isdir(toolkit):
            continue
        versions = sorted(
            (d for d in os.listdir(toolkit)
             if re.fullmatch(r"v\d+(\.\d+)*", d) and os.path.isdir(os.path.join(toolkit, d))),
            key=_version_key,
            reverse=True,
        )
        for v in versions:
            cand = os.path.join(toolkit, v)
            if os.path.isdir(os.path.join(cand, "nvvm")) and _is_trusted_cuda_dir(cand):
                return cand
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


CUDA_ROOT = ""
CUDA_DISABLED = _cuda_disabled()
if sys.platform == "win32" and not CUDA_DISABLED:
    _cuda_root = _find_cuda_root()
    CUDA_ROOT = _cuda_root

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
            # Appended, not prepended: a CUDA folder must never shadow system DLLs.
            os.environ["PATH"] = _current_path + ";" + ";".join(_dirs_to_add)

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

if _NUMBA_OK and sys.platform == "win32" and not _cuda_disabled():
    _nvvm_dll_path = globals().get("_nvvm_dll", "")

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

if _NUMBA_OK and not _cuda_disabled():
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
