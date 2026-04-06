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

import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from osgeo import gdal

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

gdal.UseExceptions()

TILE_SIZE = 512  # px — good L2/L3 cache footprint per thread
_CPU_COUNT = os.cpu_count() or 1

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


# ---------------------------------------------------------------------------
# Tile worker (called inside ThreadPoolExecutor — scipy/cv2 release the GIL)
# ---------------------------------------------------------------------------

def _filter_tile(
    tile_data: np.ndarray,
    filter_size: int,
    mode: int,
    gaussian_sigma: float,
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
                                sigma_color=gaussian_sigma)

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
            maxValue=_CPU_COUNT,
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

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                "Smoothed output",
            )
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:

        # ---- parameters ------------------------------------------------
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster_layer is None:
            raise QgsProcessingException("Invalid input raster layer.")

        preset_idx: int = self.parameterAsEnum(parameters, self.PRESET, context)

        if preset_idx != PRESET_CUSTOM:
            # Preset overrides individual controls
            _, smooth_mode, radius, gaussian_sigma = PRESETS[preset_idx]
            feedback.pushInfo(f"Preset: {PRESET_OPTIONS[preset_idx]}")
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

        # Detect bilateral backend once and log it (None for non-bilateral modes)
        bilateral_backend: str | None = None
        if smooth_mode == MODE_BILATERAL:
            from .simpli_bilateral import get_bilateral_backend
            bilateral_backend = get_bilateral_backend()
            feedback.pushInfo(f"Bilateral backend: {bilateral_backend}")
            if bilateral_backend == "CUDA":
                feedback.pushInfo(
                    "GPU path active — full raster processed in one dispatch "
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

        # ---- open with GDAL --------------------------------------------
        src_path = raster_layer.source()

        target_res = _parse_resolution(res_str)
        if target_res is not None:
            feedback.pushInfo(f"Resampling to {target_res} map units/px …")
            resampled = os.path.join(
                tempfile.gettempdir(), "simplineighbors_resampled.tif"
            )
            gdal.Warp(
                resampled, src_path,
                options=gdal.WarpOptions(
                    xRes=target_res, yRes=target_res,
                    resampleAlg=gdal.GRA_Bilinear,
                    creationOptions=["COMPRESS=LZW", "TILED=YES"],
                ),
            )
            src_path = resampled

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

        feedback.pushInfo(
            f"Raster size: {cols} × {rows} px  |  pixel size: {pixel_size:.6g} map units"
        )

        # ---- read into float32 -----------------------------------------
        feedback.pushInfo("Reading raster data …")
        data: np.ndarray = band.ReadAsArray().astype(np.float32)
        ds = None

        nodata_mask: np.ndarray | None = None
        if nodata is not None:
            nodata_mask = np.isnan(data) | (data == float(nodata))
            fill_val = float(np.nanmedian(data)) if not np.all(nodata_mask) else 0.0
            data[nodata_mask] = fill_val

        # ---- CUDA bilateral: full-raster single dispatch -------------------
        # Bypass CPU tiling entirely — entire raster lives in VRAM for the
        # duration of the kernel.  ~3–5 s on RTX 5090 for a 22k×17k raster.
        if smooth_mode == MODE_BILATERAL and bilateral_backend == "CUDA":
            feedback.pushInfo("Transferring raster to GPU …")
            from .simpli_bilateral_numba import bilateral_filter_cuda
            sigma_space = max(filter_size / 3.0, 1.0)
            output: np.ndarray = bilateral_filter_cuda(
                data, filter_size, sigma_space, gaussian_sigma
            )
            feedback.setProgress(100)

        # ---- CPU tiled processing (bilateral Numba/NumPy, Two-Pass, Median) -
        else:
            # Overlap: median radius + Gaussian effective reach (3σ)
            gaussian_reach = math.ceil(3.0 * gaussian_sigma) if smooth_mode != MODE_MEDIAN else 0
            overlap = radius + gaussian_reach

            tiles   = _build_tiles(rows, cols, TILE_SIZE, overlap)
            n_tiles = len(tiles)
            feedback.pushInfo(
                f"Processing {n_tiles} tiles "
                f"({TILE_SIZE}px core + {overlap}px overlap) on {threads} thread(s) …"
            )

            output: np.ndarray = np.empty_like(data)
            completed = 0

            with ThreadPoolExecutor(max_workers=threads) as pool:
                futures = {
                    pool.submit(
                        _filter_tile,
                        data[t["src_r0"]:t["src_r1"], t["src_c0"]:t["src_c1"]],
                        filter_size,
                        smooth_mode,
                        gaussian_sigma,
                    ): t
                    for t in tiles
                }

                for future in as_completed(futures):
                    if feedback.isCanceled():
                        pool.shutdown(wait=False, cancel_futures=True)
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

        # ---- restore nodata --------------------------------------------
        if nodata_mask is not None:
            output[nodata_mask] = float(nodata)

        # ---- write GeoTIFF ---------------------------------------------
        feedback.pushInfo(f"Writing output: {output_path}")
        driver: gdal.Driver = gdal.GetDriverByName("GTiff")
        out_ds: gdal.Dataset = driver.Create(
            output_path, cols, rows, 1, gdal.GDT_Float32,
            options=[
                "COMPRESS=LZW", "PREDICTOR=2",
                "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512",
                "BIGTIFF=IF_SAFER",
            ],
        )
        if out_ds is None:
            raise QgsProcessingException(f"Could not create: {output_path}")

        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        out_band: gdal.Band = out_ds.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(float(nodata))
        out_band.WriteArray(output)
        out_band.FlushCache()
        out_ds = None

        feedback.pushInfo("Done.")
        return {self.OUTPUT: output_path}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_resolution(value: str) -> float | None:
    v = value.strip().lower()
    if v in ("", "default"):
        return None
    try:
        f = float(v)
        return f if f > 0 else None
    except ValueError:
        return None


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
