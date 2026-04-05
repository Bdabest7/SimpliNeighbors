# SimpliNeighbors — Fast DSM Surface Smoothing Plugin for QGIS 4
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Equivalent to: r.neighbors method=median size=(2*radius+1)
# Uses scipy.ndimage.median_filter with tiled ThreadPoolExecutor for maximum
# speed without spawning new processes (avoids QGIS re-launch on Windows).

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
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

gdal.UseExceptions()

TILE_SIZE = 512  # px — good L2/L3 cache footprint per thread

_CPU_COUNT = os.cpu_count() or 1

# ---------------------------------------------------------------------------
# Thread worker — runs inside ThreadPoolExecutor (scipy releases the GIL)
# ---------------------------------------------------------------------------

def _filter_tile(tile_data: np.ndarray, filter_size: int) -> np.ndarray:
    from scipy.ndimage import median_filter
    return median_filter(tile_data.astype(np.float32), size=filter_size)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class SimpliNeighborsAlgorithm(QgsProcessingAlgorithm):

    INPUT           = "INPUT"
    RADIUS          = "RADIUS"
    THREADS         = "THREADS"
    RESOLUTION      = "RESOLUTION"   # str: "default" | float-as-string
    OUTPUT          = "OUTPUT"

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
            "Applies a fast parallel median filter to a raster layer, "
            "equivalent to GRASS r.neighbors method=median.\n\n"
            "Radius (px): half the neighborhood side. "
            "Radius R → (2R+1)×(2R+1) window.\n\n"
            "CPU cores: number of threads (scipy releases the GIL — "
            "all cores run fully parallel).\n\n"
            "Output resolution: 'default' keeps the original pixel size; "
            "enter a number (in map units/px) to resample before filtering."
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
                "Input raster (DSM)",
            )
        )

        # RADIUS — custom slider widget
        radius_param = QgsProcessingParameterNumber(
            self.RADIUS,
            "Median filter radius (px)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=6,
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

        # RESOLUTION — custom Default/Custom widget (stored as string)
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

        radius: int     = self.parameterAsInt(parameters, self.RADIUS, context)
        threads: int    = self.parameterAsInt(parameters, self.THREADS, context)
        res_str: str    = self.parameterAsString(parameters, self.RESOLUTION, context).strip()
        output_path: str = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        filter_size = 2 * radius + 1
        feedback.pushInfo(
            f"Median filter: radius={radius} px  →  window={filter_size}×{filter_size}"
        )
        feedback.pushInfo(f"Threads: {threads} / {_CPU_COUNT}")

        # ---- open with GDAL --------------------------------------------
        src_path = raster_layer.source()

        # Optional resampling step
        target_res = _parse_resolution(res_str)
        if target_res is not None:
            feedback.pushInfo(f"Resampling to {target_res} map units/px …")
            resampled = os.path.join(
                tempfile.gettempdir(), "simplineighbors_resampled.tif"
            )
            warp_opts = gdal.WarpOptions(
                xRes=target_res,
                yRes=target_res,
                resampleAlg=gdal.GRA_Bilinear,
                creationOptions=["COMPRESS=LZW", "TILED=YES"],
            )
            gdal.Warp(resampled, src_path, options=warp_opts)
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

        pixel_size = abs(geotransform[1])
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

        # ---- tiling ----------------------------------------------------
        overlap = radius
        tiles = _build_tiles(rows, cols, TILE_SIZE, overlap)
        n_tiles = len(tiles)
        feedback.pushInfo(
            f"Processing {n_tiles} tiles ({TILE_SIZE}px core + {overlap}px overlap) "
            f"on {threads} thread(s) …"
        )

        output: np.ndarray = np.empty_like(data)
        completed = 0

        # ---- parallel execution (ThreadPoolExecutor — scipy GIL-free) --
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {
                pool.submit(
                    _filter_tile,
                    data[t["src_r0"]:t["src_r1"], t["src_c0"]:t["src_c1"]],
                    filter_size,
                ): t
                for t in tiles
            }

            for future in as_completed(futures):
                if feedback.isCanceled():
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise QgsProcessingException("Cancelled by user.")

                tile = futures[future]
                result = future.result()

                r_off = tile["dst_r0"] - tile["src_r0"]
                c_off = tile["dst_c0"] - tile["src_c0"]
                h = tile["dst_r1"] - tile["dst_r0"]
                w = tile["dst_c1"] - tile["dst_c0"]
                output[tile["dst_r0"]:tile["dst_r1"], tile["dst_c0"]:tile["dst_c1"]] = (
                    result[r_off: r_off + h, c_off: c_off + w]
                )

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
    """Return float if a custom resolution was requested, else None."""
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
    n_row = math.ceil(rows / tile_size)
    n_col = math.ceil(cols / tile_size)
    for ri in range(n_row):
        for ci in range(n_col):
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
