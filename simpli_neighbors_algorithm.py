# SimpliNeighbors — Fast DSM Surface Smoothing Plugin for QGIS 4
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Optimised for golf course fairway / green DTM smoothing:
#   — Removes vegetation spikes and scanner noise (Pass 1: Median)
#   — Preserves sharp topographic breaks (bunker lips, green collars)
#   — Eliminates terracing on gentle slopes (Pass 2: Gaussian / Bilateral)
#
# Smoothing modes:
#   0 — Two-Pass: large Median (despiking) → small Gaussian (anti-aliasing)
#   1 — Median only: classic r.neighbors method=median equivalent
#   2 — Bilateral: bundled pure-NumPy edge-preserving filter (no OpenCV needed)
#
# Processing uses tiled ThreadPoolExecutor so all CPU cores run in parallel
# without spawning new processes (avoids QGIS re-launch on Windows).

from __future__ import annotations

import contextlib
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from osgeo import gdal, ogr, osr

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)


TILE_SIZE = 512  # px — good L2/L3 cache footprint per thread
_CPU_COUNT = os.cpu_count() or 1
# THREADS is bounded by sanity, not by this machine: the same saved parameters
# (e.g. THREADS=64) must be accepted on a machine with fewer cores too.
MAX_THREADS = 1024
# Upper limit of 512 px tiles in one run (about 26 gigapixels); raise with
# SIMPLINEIGHBORS_MAX_TILES. The Jackson CC DTM is about 1,200 tiles.
DEFAULT_MAX_TILES = 100_000

# Enum indices for SMOOTH_MODE
MODE_TWO_PASS  = 0
MODE_MEDIAN    = 1
MODE_BILATERAL = 2

SMOOTH_MODE_OPTIONS = [
    "Two-Pass: Median + Gaussian  (recommended — no terracing)",
    "Median only  (fast despiking, may terrace on gentle slopes)",
    "Bilateral  (best quality — preserves bunker lips, smooths fairway slopes)",
]

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# Each preset is (label, smooth_mode, radius, gaussian_sigma).
# sigma meaning depends on mode:
#   Two-Pass  → gaussian_sigma is anti-aliasing px
#   Bilateral → gaussian_sigma is sigma_color in map units (e.g. metres)

PRESET_CUSTOM = 0

PRESETS = [
    # index 0 — sentinel, user controls all parameters manually
    ("Custom  (use values below)", None, None, None),

    # Whole-course at 7 cm/px — the primary use case.
    # Bilateral sigma_color=0.10 m: fairway slopes change <7 cm over 49 cm
    # window → smooth blending everywhere; collar breaks (≥8 cm) and bunker
    # lips (≥20 cm) exceed the threshold → hard-preserved.
    ("Whole Course — Standard  (Bilateral R=16, σ=0.10 m)",
     MODE_BILATERAL, 16, 0.10),

    # Heavier vegetation (rough, tree drip-lines).  Larger median window
    # kills tall shrub hits; Two-Pass Gaussian removes resulting terracing.
    ("Whole Course — Heavy Vegetation  (Two-Pass R=5, σ=1.5 px)",
     MODE_TWO_PASS, 5, 1.5),

    # Greens-focused: tighter sigma_color preserves even subtle collar
    # transitions (≥8 cm).
    ("Greens / Approaches  (Bilateral R=3, σ=0.08 m)",
     MODE_BILATERAL, 3, 0.08),

    # Fairways only, no complex edge geometry needed.
    ("Fairways / Tees  (Two-Pass R=3, σ=1.5 px)",
     MODE_TWO_PASS, 3, 1.5),
]

PRESET_OPTIONS = [p[0] for p in PRESETS]

# Bilateral backend choice (advanced parameter; Auto = best available, as before)
BACKEND_OPTIONS = ["Auto (best available)", "CUDA (GPU)", "Numba CPU", "NumPy"]
BACKEND_OPTION_VALUES = ["auto", "cuda", "numba", "numpy"]


# ---------------------------------------------------------------------------
# Tile worker (called inside ThreadPoolExecutor — scipy/cv2 release the GIL)
# ---------------------------------------------------------------------------

def _filter_tile(
    tile_data: np.ndarray,
    filter_size: int,
    mode: int,
    gaussian_sigma: float,
    backend: str | None = None,
) -> np.ndarray:
    """
    Apply the selected smoothing algorithm to one tile.

    Two-Pass (mode 0) — recommended for golf course DTMs
    -----------------------------------------------------
    Pass 1 — median_filter(size=filter_size):
        Kills vegetation spikes and scanner noise. The median is inherently
        robust to outliers so sharp edges (bunker lips, green collars, cart
        path kerbs) are preserved. On gentle fairway / green slopes this
        introduces micro-terracing.

    Pass 2 — gaussian_filter(sigma=gaussian_sigma):
        Narrow Gaussian acts as fine-grit sandpaper — dissolves the
        micro-steps back into a continuous gradient without being wide
        enough to round off the macroscopic edges saved by Pass 1.

    Median only (mode 1)
    --------------------
    Diagnostic / legacy mode. Good when you only need spike removal and
    terracing is acceptable (very flat terrain, coarse GSD).

    Bilateral (mode 2) — best for complex green / bunker geometry
    -------------------------------------------------------------
    Bundled pure-NumPy implementation (no OpenCV required).
    Weights each neighbour by both spatial distance AND elevation difference.
    Smooths continuous slopes (fairway undulations, green bowls) while
    mathematically halting at intensity edges (bunker lips, green perimeter
    breaks). Ideal when sigma_color is tuned to the expected lip height
    (e.g. 0.15 m for a 15 cm bunker lip on a metre-scale DTM).
    """
    data = tile_data.astype(np.float32)

    if mode == MODE_MEDIAN:
        from scipy.ndimage import median_filter
        return median_filter(data, size=filter_size)

    if mode == MODE_BILATERAL:
        from .simpli_bilateral import bilateral_filter
        # sigma_space: spatial reach — use 1/3 of window so edge weights taper
        sigma_space = max(filter_size / 3.0, 1.0)
        # sigma_color: elevation range — gaussian_sigma repurposed as map units
        return bilateral_filter(data, filter_size,
                                sigma_space=sigma_space,
                                sigma_color=gaussian_sigma,
                                backend=backend)

    # Two-Pass (default)
    from scipy.ndimage import median_filter, gaussian_filter
    despiked = median_filter(data, size=filter_size)
    return gaussian_filter(despiked, sigma=gaussian_sigma)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class SimpliNeighborsAlgorithm(QgsProcessingAlgorithm):

    INPUT          = "INPUT"
    PRESET         = "PRESET"
    SMOOTH_MODE    = "SMOOTH_MODE"
    RADIUS         = "RADIUS"
    GAUSSIAN_SIGMA = "GAUSSIAN_SIGMA"
    THREADS        = "THREADS"
    RESOLUTION     = "RESOLUTION"
    BACKEND        = "BACKEND"
    OUTPUT         = "OUTPUT"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "surfacesmoothing"

    def displayName(self) -> str:
        return "Surface Smoothing (Median Filter)"

    def group(self) -> str:
        return "Raster"

    def groupId(self) -> str:
        return "raster"

    def shortHelpString(self) -> str:
        return (
            "<b>Designed for golf course fairway and green DTM smoothing.</b>"
            "<br><br>"
            "<b>Two-Pass (recommended)</b>: Median despiking followed by a "
            "narrow Gaussian. Removes grass / vegetation hits and scanner "
            "noise; eliminates terracing on gentle fairway slopes; preserves "
            "bunker lips and green collars.<br><br>"
            "<b>Bilateral (best quality)</b>: Bundled pure-NumPy "
            "edge-preserving filter — no OpenCV needed. Smooths continuous "
            "gradients (green bowls, fairway undulations) while halting at "
            "intensity edges (bunker lips, collar breaks). Set "
            "<i>Gaussian sigma</i> to the expected lip height in map units "
            "(e.g. 0.15 for a 15 cm bunker lip on a metre-scale DTM).<br><br>"
            "<b>Median only</b>: Classic r.neighbors equivalent. Fast spike "
            "removal; may terrace on gentle slopes.<br><br>"
            "<b>Radius</b>: half the median window (R → (2R+1)² px). "
            "At 1 cm/px GSD: radius 6 = 13×13 cm window — enough to remove "
            "grass blades while keeping fairway microrelief.<br><br>"
            "<b>Gaussian sigma</b>: Two-Pass anti-aliasing width in pixels "
            "(1–2 px typical). Bilateral: intensity range threshold in map "
            "units — set near the smallest edge height you want to preserve."
        )

    def createInstance(self) -> "SimpliNeighborsAlgorithm":
        return SimpliNeighborsAlgorithm()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def initAlgorithm(self, config=None) -> None:
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                "Input raster (golf course DTM/DSM)",
            )
        )

        # PRESET — shown first; overrides all parameters below when not Custom
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PRESET,
                "Preset  (overrides settings below when not Custom)",
                options=PRESET_OPTIONS,
                defaultValue=1,   # "Whole Course — Standard" is the default
            )
        )

        # Smoothing mode — standard enum dropdown
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SMOOTH_MODE,
                "Smoothing algorithm  [ignored when Preset ≠ Custom]",
                options=SMOOTH_MODE_OPTIONS,
                defaultValue=MODE_BILATERAL,
            )
        )

        # RADIUS — custom slider widget
        radius_param = QgsProcessingParameterNumber(
            self.RADIUS,
            "Median filter radius (px)  [ignored when Preset ≠ Custom]",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=16,
            minValue=1,
            maxValue=50,
        )
        try:
            from .simpli_neighbors_widget import SimpliNeighborsRadiusWrapper
            radius_param.setMetadata(
                {"widget_wrapper": {"class": SimpliNeighborsRadiusWrapper}}
            )
        except Exception:
            pass
        self.addParameter(radius_param)

        # GAUSSIAN_SIGMA — anti-aliasing sigma / bilateral intensity range
        self.addParameter(
            QgsProcessingParameterNumber(
                self.GAUSSIAN_SIGMA,
                "Gaussian sigma  [Two-Pass: anti-aliasing px | Bilateral: edge threshold in map units]  [ignored when Preset ≠ Custom]",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.10,
                minValue=0.01,
                maxValue=10.0,
            )
        )

        # THREADS — custom slider widget
        threads_param = QgsProcessingParameterNumber(
            self.THREADS,
            "CPU cores",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=_CPU_COUNT,
            minValue=1,
            maxValue=MAX_THREADS,
        )
        try:
            from .simpli_neighbors_widget import SimpliNeighborsCoresWrapper
            threads_param.setMetadata(
                {"widget_wrapper": {"class": SimpliNeighborsCoresWrapper}}
            )
        except Exception:
            pass
        self.addParameter(threads_param)

        # RESOLUTION — custom Default/Custom widget
        res_param = QgsProcessingParameterString(
            self.RESOLUTION,
            "Output resolution",
            defaultValue="default",
            optional=False,
        )
        try:
            from .simpli_neighbors_widget import SimpliNeighborsResolutionWrapper
            res_param.setMetadata(
                {"widget_wrapper": {"class": SimpliNeighborsResolutionWrapper}}
            )
        except Exception:
            pass
        self.addParameter(res_param)

        # BACKEND — advanced, so the dialog looks as before; the automation
        # pins it so results do not depend on what an account happens to have.
        backend_param = QgsProcessingParameterEnum(
            self.BACKEND,
            "Bilateral backend  [Bilateral mode only; a pinned backend that is missing is an error]",
            options=BACKEND_OPTIONS,
            defaultValue=0,
        )
        try:
            backend_param.setFlags(
                backend_param.flags() | QgsProcessingParameterDefinition.Flag.Advanced)
        except AttributeError:  # QGIS 3 spelling
            backend_param.setFlags(
                backend_param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(backend_param)

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                "Smoothed output",
            )
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def checkParameterValues(self, parameters, context):
        # Catch an unreadable RESOLUTION before anything runs (finding 13)
        try:
            _parse_resolution(self.parameterAsString(parameters, self.RESOLUTION, context))
        except ValueError as e:
            return False, str(e)
        return super().checkParameterValues(parameters, context)


    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        # GDAL exceptions are switched on for this run only and the previous
        # setting is restored afterwards (finding 15).
        with _gdal_exceptions():
            return self._process(parameters, context, feedback)

    def _process(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:

        # ---- parameters ------------------------------------------------
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster_layer is None or not raster_layer.isValid():
            raise QgsProcessingException("Invalid input raster layer.")

        preset_idx: int = self.parameterAsEnum(parameters, self.PRESET, context)
        use_preset, preset_note = resolve_settings(parameters, preset_idx)
        if preset_note:
            feedback.pushWarning(preset_note)

        if use_preset:
            # Preset overrides individual controls
            _, smooth_mode, radius, gaussian_sigma = PRESETS[preset_idx]
            feedback.pushInfo(f"Preset: {PRESET_OPTIONS[preset_idx]}")
            # The dialog always sends the individual values too. Say so only
            # when they differ from the preset, which then overrides them.
            if parameters.get(self.PRESET) is not None:
                given = {}
                if parameters.get(self.SMOOTH_MODE) is not None:
                    given["smoothing algorithm"] = (self.parameterAsEnum(parameters, self.SMOOTH_MODE, context), smooth_mode)
                if parameters.get(self.RADIUS) is not None:
                    given["radius"] = (self.parameterAsInt(parameters, self.RADIUS, context), radius)
                if parameters.get(self.GAUSSIAN_SIGMA) is not None:
                    given["gaussian sigma"] = (self.parameterAsDouble(parameters, self.GAUSSIAN_SIGMA, context), gaussian_sigma)
                diff = [f"{k} {a} (preset: {b})" for k, (a, b) in given.items() if a != b]
                if diff:
                    feedback.pushWarning(
                        "The preset overrides these explicit settings: " + "; ".join(diff) +
                        ". Set Preset to Custom to use your own values.")
        else:
            smooth_mode: int      = self.parameterAsEnum(parameters, self.SMOOTH_MODE, context)
            radius: int           = self.parameterAsInt(parameters, self.RADIUS, context)
            gaussian_sigma: float = self.parameterAsDouble(parameters, self.GAUSSIAN_SIGMA, context)
            feedback.pushInfo("Preset: Custom")

        threads: int     = self.parameterAsInt(parameters, self.THREADS, context)
        res_str: str     = self.parameterAsString(parameters, self.RESOLUTION, context).strip()
        output_path: str = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        filter_size = 2 * radius + 1
        mode_label = SMOOTH_MODE_OPTIONS[smooth_mode].split("(")[0].strip()

        feedback.pushInfo(f"Mode: {mode_label}")
        feedback.pushInfo(
            f"Median radius={radius} px  →  window={filter_size}×{filter_size}"
        )
        if smooth_mode in (MODE_TWO_PASS, MODE_BILATERAL):
            feedback.pushInfo(f"Gaussian sigma / sigmaColor = {gaussian_sigma}")
        feedback.pushInfo(f"Threads: {threads} / {_CPU_COUNT}")
        if threads > _CPU_COUNT:
            feedback.pushWarning(
                f"{threads} threads requested but this machine has {_CPU_COUNT} cores")

        # ---- dependencies: fail early, naming what is missing (finding 14)
        from . import simpli_deps
        try:
            if smooth_mode == MODE_BILATERAL:
                # scipy is not needed by the bilateral filter; numba only if pinned/used
                simpli_deps.require_for(None, mode_needs_scipy=False)
            else:
                simpli_deps.require_for(None, mode_needs_scipy=True)
        except ImportError as e:
            raise QgsProcessingException(str(e))

        # Choose the bilateral backend, honouring a pin, and log it (finding 8).
        # The BACKEND parameter wins; when it is Auto, SIMPLINEIGHBORS_BACKEND
        # may pin one. A pinned backend that is missing is an error.
        bilateral_backend: str | None = None
        if smooth_mode == MODE_BILATERAL:
            from . import simpli_bilateral as _sb
            backend_idx = self.parameterAsEnum(parameters, self.BACKEND, context) \
                if parameters.get(self.BACKEND) is not None else 0
            requested = BACKEND_OPTION_VALUES[backend_idx]
            source = "BACKEND parameter"
            if requested == "auto" and os.environ.get("SIMPLINEIGHBORS_BACKEND", "").strip():
                requested = os.environ["SIMPLINEIGHBORS_BACKEND"]
                source = "SIMPLINEIGHBORS_BACKEND"
            try:
                bilateral_backend = _sb.resolve_backend(requested)
            except _sb.BackendUnavailable as e:
                raise QgsProcessingException(str(e))
            pinned = _sb.normalize_backend_name(requested) is not None
            feedback.pushInfo(
                f"Bilateral backend: {bilateral_backend}"
                + (f"  (pinned by {source})" if pinned else "  (auto-selected)")
            )
            feedback.pushInfo(f"Backend details: {_sb.describe_backend(bilateral_backend)}")
            if bilateral_backend == "CUDA":
                feedback.pushInfo(
                    "GPU path active — one dispatch per row band "
                    "(no CPU tiling).  First run compiles CUDA kernel (~10–30 s)."
                )
            elif bilateral_backend == "Numba CPU":
                feedback.pushInfo(
                    "Numba CPU path active — fastmath JIT per tile.  "
                    "First run compiles kernel (~5–10 s)."
                )
            else:
                feedback.pushInfo(
                    "NumPy fallback active.  Install numba for faster processing: "
                    "pip install numba  (in OSGeo4W shell)"
                )
        else:
            feedback.pushInfo("Backend: scipy (only the Bilateral mode uses CUDA / Numba / NumPy)")

        # ---- open with GDAL --------------------------------------------
        src_path = raster_layer.source()
        try:
            target_res = _parse_resolution(res_str)
        except ValueError as e:
            raise QgsProcessingException(str(e))
        if target_res is not None:
            feedback.pushInfo(f"Resampling to {target_res} map units/px …")
            # Resample into an in-memory dataset: no file at a fixed path in
            # %TEMP% (concurrent runs collided on it and it was never removed),
            # nothing to clean up, and it disappears with the run (finding 3).
            ds: gdal.Dataset = gdal.Warp(
                "", src_path,
                options=gdal.WarpOptions(
                    format="MEM",
                    xRes=target_res, yRes=target_res,
                    resampleAlg=gdal.GRA_Bilinear,
                ),
            )
            if ds is None:
                raise QgsProcessingException(f"GDAL could not resample: {src_path}")
        else:
            ds: gdal.Dataset = gdal.Open(src_path, gdal.GA_ReadOnly)
            if ds is None:
                raise QgsProcessingException(f"GDAL could not open: {src_path}")

        band: gdal.Band = ds.GetRasterBand(1)
        cols: int       = ds.RasterXSize
        rows: int       = ds.RasterYSize
        geotransform    = ds.GetGeoTransform()
        projection      = ds.GetProjection()
        nodata          = band.GetNoDataValue()
        pixel_size      = abs(geotransform[1])

        # ---- sanity limits (finding 13) -------------------------------
        if cols < 1 or rows < 1:
            raise QgsProcessingException(f"Input raster has no pixels ({cols} x {rows})")
        n_tiles_total = math.ceil(rows / TILE_SIZE) * math.ceil(cols / TILE_SIZE)
        max_tiles = _max_tiles()
        if n_tiles_total > max_tiles:
            raise QgsProcessingException(
                f"{n_tiles_total} tiles of {TILE_SIZE} px needed ({cols} x {rows} px), more "
                f"than the limit of {max_tiles} per run. Check the input and the output "
                f"resolution (the limit can be raised with SIMPLINEIGHBORS_MAX_TILES)."
            )
        if filter_size > min(rows, cols):
            raise QgsProcessingException(
                f"Filter window {filter_size} x {filter_size} px is larger than the raster "
                f"({cols} x {rows} px); lower the radius"
            )

        feedback.pushInfo(
            f"Raster size: {cols} × {rows} px  |  pixel size: {pixel_size:.6g} map units"
        )

        # ---- plan the row bands (findings 5 and 10) ----------------------
        use_cuda = smooth_mode == MODE_BILATERAL and bilateral_backend == "CUDA"
        sigma_space = max(filter_size / 3.0, 1.0)
        if use_cuda:
            overlap = filter_size // 2          # the kernel's reach
            tiles_all = None
        else:
            # Overlap: median radius + Gaussian effective reach (3σ)
            gaussian_reach = math.ceil(3.0 * gaussian_sigma) if smooth_mode != MODE_MEDIAN else 0
            overlap = radius + gaussian_reach
            tiles_all = _build_tiles(rows, cols, TILE_SIZE, overlap)
        core_rows = _band_core_rows(cols, filter_size, use_cuda)
        bands = _plan_bands(rows, core_rows, overlap)
        feedback.pushInfo(
            f"Processing {len(bands)} row band(s) of up to {core_rows} rows "
            f"(+{overlap} rows overlap each side)"
            + ("" if use_cuda else f", {len(tiles_all)} tiles of {TILE_SIZE}px on {threads} thread(s)")
            + " …"
        )

        def read_rows(r0: int, r1: int) -> np.ndarray:
            return band.ReadAsArray(0, r0, cols, r1 - r0).astype(np.float32)

        # ---- nodata: one global fill value, as before --------------------
        fill_val = None
        progress_start = 0
        if nodata is not None:
            feedback.pushInfo("Scanning for nodata …")
            fill_val = _global_fill_value(read_rows, rows, float(nodata), feedback, 0, 15)
            progress_start = 15
            if fill_val is None:
                feedback.pushInfo("No nodata pixels.")
            else:
                feedback.pushInfo(f"Nodata pixels are filled with {fill_val:.6g} while filtering "
                                  f"and restored afterwards.")

        # ---- filter band by band and write each core as soon as it is done
        out = AtomicGeoTiff(output_path, cols, rows, geotransform, projection, nodata)
        feedback.pushInfo(f"Writing output: {output_path}")
        try:
            for i, b in enumerate(bands):
                if feedback.isCanceled():
                    raise QgsProcessingException("Cancelled by user.")
                data = read_rows(b["s0"], b["s1"])
                nodata_mask = None
                if nodata is not None and fill_val is not None:
                    nodata_mask = np.isnan(data) | (data == float(nodata))
                    data[nodata_mask] = fill_val

                if use_cuda:
                    # One kernel launch per band: far below the 2 s GPU timeout.
                    from .simpli_bilateral_numba import bilateral_filter_cuda
                    result = bilateral_filter_cuda(data, filter_size, sigma_space, gaussian_sigma)
                else:
                    result = np.empty_like(data)
                    _filter_tiles(
                        data, result, _tiles_for_band(tiles_all, b), filter_size, smooth_mode,
                        gaussian_sigma, threads, feedback, backend=bilateral_backend,
                    )

                lo, hi = b["r0"] - b["s0"], b["r1"] - b["s0"]
                core = result[lo:hi]
                if nodata_mask is not None:
                    core[nodata_mask[lo:hi]] = float(nodata)
                out.write(core, b["r0"])
                del data, result, core, nodata_mask
                feedback.pushInfo(f"  band {i + 1}/{len(bands)} done (rows {b['r0']}–{b['r1']})")
                feedback.setProgress(progress_start + (100 - progress_start) * (i + 1) / len(bands))
            ds = None
            out.commit()
        except BaseException:
            out.abandon()
            feedback.reportError(
                f"Output not complete. The unfinished file {out.partial_path} was left; "
                f"{output_path} was not replaced.")
            raise

        feedback.pushInfo("Done.")
        return {self.OUTPUT: output_path}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_resolution(value: str) -> float | None:
    """None for blank / "default"; a positive finite number otherwise.

    Anything else is an error: an unreadable value used to mean "default"
    without a word, so a typo silently produced the wrong resolution.
    """
    v = value.strip().lower()
    if v in ("", "default"):
        return None
    try:
        f = float(v)
    except ValueError:
        raise ValueError(
            f"Output resolution {value!r} is not a number; use \"default\" or a "
            f"positive number of map units per pixel")
    if not math.isfinite(f) or f <= 0:
        raise ValueError(
            f"Output resolution {value!r} must be a positive number of map units per pixel")
    return f


def _build_tiles(rows: int, cols: int, tile_size: int, overlap: int) -> list[dict]:
    tiles = []
    for ri in range(math.ceil(rows / tile_size)):
        for ci in range(math.ceil(cols / tile_size)):
            dst_r0 = ri * tile_size
            dst_r1 = min(dst_r0 + tile_size, rows)
            dst_c0 = ci * tile_size
            dst_c1 = min(dst_c0 + tile_size, cols)
            tiles.append({
                "dst_r0": dst_r0, "dst_r1": dst_r1,
                "dst_c0": dst_c0, "dst_c1": dst_c1,
                "src_r0": max(dst_r0 - overlap, 0),
                "src_r1": min(dst_r1 + overlap, rows),
                "src_c0": max(dst_c0 - overlap, 0),
                "src_c1": min(dst_c1 + overlap, cols),
            })
    return tiles


def resolve_settings(parameters: dict, preset_idx: int) -> tuple[bool, str | None]:
    """Decide whether the PRESET or the explicit mode/radius/sigma values apply.

    Returns (use_preset, note). The note, if any, is for the log.

    - PRESET passed (as the dialog and the automation always do): unchanged
      from before. A non-Custom preset overrides the individual settings; if the
      explicit values differ from the preset's, the log says so.
    - PRESET not passed but SMOOTH_MODE / RADIUS / GAUSSIAN_SIGMA were: the
      explicit values win (treated as Custom). Before, the PRESET default of 1
      silently replaced them.
    - Nothing passed: the default preset 1, as before.
    """
    explicit = [k for k in (SimpliNeighborsAlgorithm.SMOOTH_MODE,
                            SimpliNeighborsAlgorithm.RADIUS,
                            SimpliNeighborsAlgorithm.GAUSSIAN_SIGMA)
                if parameters.get(k) is not None]
    preset_given = parameters.get(SimpliNeighborsAlgorithm.PRESET) is not None
    if not preset_given and explicit:
        return False, ("PRESET was not given, so the explicit " + ", ".join(explicit) +
                       " apply (Custom) instead of the default preset")
    return preset_idx != PRESET_CUSTOM, None


def _filter_tiles(
    data: np.ndarray,
    output: np.ndarray,
    tiles: list,
    filter_size: int,
    smooth_mode: int,
    gaussian_sigma: float,
    threads: int,
    feedback,
    backend: str | None = None,
) -> None:
    """Filter every tile of `data` on a thread pool and copy the cores into `output`."""
    n_tiles = len(tiles)
    completed = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(
                _filter_tile,
                data[t["src_r0"]:t["src_r1"], t["src_c0"]:t["src_c1"]],
                filter_size,
                smooth_mode,
                gaussian_sigma,
                backend,
            ): t
            for t in tiles
        }
        try:
            for future in as_completed(futures):
                if feedback.isCanceled():
                    raise QgsProcessingException("Cancelled by user.")

                tile   = futures[future]
                result = future.result()

                r_off = tile["dst_r0"] - tile["src_r0"]
                c_off = tile["dst_c0"] - tile["src_c0"]
                h     = tile["dst_r1"] - tile["dst_r0"]
                w     = tile["dst_c1"] - tile["dst_c0"]
                output[
                    tile["dst_r0"]:tile["dst_r1"],
                    tile["dst_c0"]:tile["dst_c1"],
                ] = result[r_off: r_off + h, c_off: c_off + w]

                completed += 1
                feedback.setProgress(int(completed / n_tiles * 100))
        except BaseException:
            # A failed tile (or a cancel) must not wait for every
            # remaining tile: drop the queued ones, let only the
            # tiles already running finish, then re-raise (finding 7).
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def _max_tiles() -> int:
    """Tile limit per run: DEFAULT_MAX_TILES, or SIMPLINEIGHBORS_MAX_TILES if set.
    An unreadable value is an error, not ignored."""
    raw = os.environ.get("SIMPLINEIGHBORS_MAX_TILES", "").strip()
    if not raw:
        return DEFAULT_MAX_TILES
    try:
        value = int(raw)
    except ValueError:
        raise QgsProcessingException(f"SIMPLINEIGHBORS_MAX_TILES is not an integer: {raw!r}")
    if value < 1:
        raise QgsProcessingException(f"SIMPLINEIGHBORS_MAX_TILES must be at least 1, got {value}")
    return value


@contextlib.contextmanager
def _gdal_exceptions():
    """Turn GDAL (and OGR/OSR) exceptions on for a run, then restore what was set before."""
    mods = (gdal, ogr, osr)
    prev = {m: bool(m.GetUseExceptions()) if hasattr(m, "GetUseExceptions") else False for m in mods}
    gdal.UseExceptions()
    try:
        yield
    finally:
        for m, was_on in prev.items():
            if was_on:
                m.UseExceptions()
            else:
                m.DontUseExceptions()


class AtomicGeoTiff:
    """Write the output GeoTIFF to '<output>.partial' and move it to the real
    name only when it is complete (finding 4).

    A cancelled, failed or killed run leaves at most '<output>.partial' (never
    deleted by the plugin, rewritten by the next run of the same job); nothing
    at the output path looks complete unless it is. An older file at the output
    path is replaced only by the finished one.
    """

    CREATION_OPTIONS = [
        "COMPRESS=LZW", "PREDICTOR=2",
        "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512",
        "BIGTIFF=IF_SAFER",
    ]

    def __init__(self, output_path: str, cols: int, rows: int, geotransform,
                 projection: str, nodata):
        self.output_path = output_path
        self.partial_path = output_path + ".partial"
        driver: gdal.Driver = gdal.GetDriverByName("GTiff")
        self._ds = driver.Create(
            self.partial_path, cols, rows, 1, gdal.GDT_Float32,
            options=self.CREATION_OPTIONS,
        )
        if self._ds is None:
            raise QgsProcessingException(f"Could not create: {self.partial_path}")
        self._ds.SetGeoTransform(geotransform)
        self._ds.SetProjection(projection)
        self._band: gdal.Band = self._ds.GetRasterBand(1)
        if nodata is not None:
            self._band.SetNoDataValue(float(nodata))

    def write(self, array: np.ndarray, yoff: int = 0) -> None:
        self._band.WriteArray(array, 0, yoff)

    def commit(self) -> None:
        """Close the partial file and move it to the output path."""
        self._band.FlushCache()
        self._band = None
        self._ds = None  # closes and flushes the file
        try:
            os.replace(self.partial_path, self.output_path)
        except OSError as e:
            raise QgsProcessingException(
                f"The result is complete in {self.partial_path} but could not be moved to "
                f"{self.output_path} (is the old file open in another program?): {e}")

    def abandon(self) -> None:
        """Close the partial file after a failure; it is left on disk, incomplete."""
        self._band = None
        self._ds = None


# ---------------------------------------------------------------------------
# Row bands (findings 5 and 10)
#
# The raster is processed in horizontal bands of whole 512-px tile rows. Each
# band is read with `overlap` extra rows above and below, filtered, and only
# its core rows are written. Because band boundaries fall on the tile grid and
# the overlap is at least what each filter needs, every output pixel sees
# exactly the same input as in a whole-raster run: the output is identical
# (tested), but only one band is in memory instead of 4-6 copies of the raster,
# progress is reported after every band, and a CUDA launch covers a band, not
# the whole raster, so it stays far below Windows' 2 s GPU timeout.
# ---------------------------------------------------------------------------

_BAND_PIXELS_CPU = 2 ** 25      # about 134 MB of float32 per band core
_BAND_PIXELS_CUDA = 2 ** 26
_CUDA_TAPS_PER_LAUNCH = 1.5e11  # measured 300-900 Gtap/s on the RTX 5090: well under 1 s


def _band_core_rows(cols: int, filter_size: int, cuda: bool) -> int:
    """Core rows per band: a multiple of TILE_SIZE (override for tests and
    tuning with SIMPLINEIGHBORS_BAND_ROWS)."""
    raw = os.environ.get("SIMPLINEIGHBORS_BAND_ROWS", "").strip()
    if raw:
        try:
            rows = int(raw)
        except ValueError:
            raise QgsProcessingException(f"SIMPLINEIGHBORS_BAND_ROWS is not an integer: {raw!r}")
        if rows < 1:
            raise QgsProcessingException("SIMPLINEIGHBORS_BAND_ROWS must be at least 1")
    else:
        rows = (_BAND_PIXELS_CUDA if cuda else _BAND_PIXELS_CPU) // max(cols, 1)
        if cuda:
            rows = min(rows, int(_CUDA_TAPS_PER_LAUNCH / (max(cols, 1) * filter_size * filter_size)))
    return max(TILE_SIZE, (rows // TILE_SIZE) * TILE_SIZE)


def _plan_bands(rows: int, core_rows: int, overlap: int) -> list[dict]:
    """Bands as dicts: core rows [r0, r1) and the rows to read [s0, s1)."""
    bands = []
    for r0 in range(0, rows, core_rows):
        r1 = min(r0 + core_rows, rows)
        bands.append({"r0": r0, "r1": r1,
                      "s0": max(r0 - overlap, 0), "s1": min(r1 + overlap, rows)})
    return bands


def _tiles_for_band(tiles: list[dict], band: dict) -> list[dict]:
    """The global tiles whose core lies in the band, with rows relative to band['s0']."""
    s0 = band["s0"]
    out = []
    for t in tiles:
        if t["dst_r0"] >= band["r0"] and t["dst_r1"] <= band["r1"]:
            u = dict(t)
            for k in ("dst_r0", "dst_r1", "src_r0", "src_r1"):
                u[k] = t[k] - s0
            out.append(u)
    return out


def _float_keys(a: np.ndarray) -> np.ndarray:
    """Map float32 values to uint32 keys that sort like the floats."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return np.where(u & np.uint32(0x80000000), ~u, u | np.uint32(0x80000000)).astype(np.uint32)


def _keys_to_float(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.uint32)
    u = np.where(keys & np.uint32(0x80000000), keys & np.uint32(0x7FFFFFFF), ~keys).astype(np.uint32)
    return u.view(np.float32)


def _global_fill_value(read_rows, rows: int, nodata: float, feedback=None,
                       progress_lo: int = 0, progress_hi: int = 0, stripe: int = 1024):
    """The value that replaces nodata pixels before filtering, computed exactly
    as the whole-raster code did but without holding the raster.

    The earlier code used np.nanmedian of the whole float32 array *including*
    the nodata-valued pixels (only NaN was left out), and 0.0 when every pixel
    is nodata/NaN. That is kept exactly, so results do not change. (It means a
    large nodata share pulls the fill value down; see REVIEW-FIXES.md.)

    The median is found exactly in two streaming passes: a 65,536-bin histogram
    of the order-preserving float keys' top 16 bits, then the few values in the
    bin(s) holding the middle rank.

    read_rows(r0, r1) returns rows [r0, r1) as float32. Returns the fill value,
    or None when no pixel is nodata/NaN (nothing to fill).
    """
    def stripes():
        for r0 in range(0, rows, stripe):
            yield r0, read_rows(r0, min(r0 + stripe, rows))

    hist = np.zeros(65536, dtype=np.int64)
    n_masked = 0
    n_total = 0
    n_stripes = max(1, math.ceil(rows / stripe))
    done = 0
    for _r0, a in stripes():
        if feedback is not None and feedback.isCanceled():
            raise QgsProcessingException("Cancelled by user.")
        nan = np.isnan(a)
        n_masked += int((nan | (a == nodata)).sum())
        keys = _float_keys(a[~nan])
        hist += np.bincount(keys >> np.uint32(16), minlength=65536)
        n_total += a.size
        done += 1
        if feedback is not None and progress_hi > progress_lo:
            feedback.setProgress(progress_lo + (progress_hi - progress_lo) * 0.5 * done / n_stripes)
    if n_masked == 0:
        return None
    if n_masked == n_total:
        return 0.0
    n = int(hist.sum())  # non-NaN pixels
    ranks = [(n - 1) // 2] if n % 2 else [n // 2 - 1, n // 2]
    cum = np.cumsum(hist)
    bins = {r: int(np.searchsorted(cum, r, side="right")) for r in ranks}
    wanted = sorted(set(bins.values()))
    collected = {b: [] for b in wanted}
    done = 0
    for _r0, a in stripes():
        if feedback is not None and feedback.isCanceled():
            raise QgsProcessingException("Cancelled by user.")
        keys = _float_keys(a[~np.isnan(a)])
        top = keys >> np.uint32(16)
        for b in wanted:
            sel = keys[top == np.uint32(b)]
            if sel.size:
                collected[b].append(sel)
        done += 1
        if feedback is not None and progress_hi > progress_lo:
            feedback.setProgress(progress_lo + (progress_hi - progress_lo) * (0.5 + 0.5 * done / n_stripes))
    vals = []
    for r in ranks:
        b = bins[r]
        arr = np.sort(np.concatenate(collected[b]))
        vals.append(float(_keys_to_float(arr[r - int(cum[b] - hist[b])])))
    if len(vals) == 1:
        return float(vals[0])
    return float(np.mean(np.array(vals, dtype=np.float32)))  # what np.median does for two values
