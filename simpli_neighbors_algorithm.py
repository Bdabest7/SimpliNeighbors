# SimpliNeighbors — Fast DSM Surface Smoothing Plugin for QGIS 4
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Equivalent to: r.neighbors method=median size=(2*radius+1)
# Uses scipy.ndimage.median_filter with tiled multiprocessing for maximum speed.

from __future__ import annotations

import math
import multiprocessing
import os

import numpy as np
from osgeo import gdal

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# Module-level worker — must be top-level for multiprocessing pickling
# ---------------------------------------------------------------------------

def _process_tile(args: tuple) -> np.ndarray:
    """Apply median filter to a single tile (runs in worker process)."""
    tile_data, filter_size = args
    from scipy.ndimage import median_filter  # import inside worker
    return median_filter(tile_data.astype(np.float32), size=filter_size)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

TILE_SIZE = 512  # pixels — good balance of L2 cache use vs parallelism


class SimpliNeighborsAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    RADIUS = "RADIUS"
    OUTPUT = "OUTPUT"

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
            "equivalent to GRASS r.neighbors with method=median.\n\n"
            "Radius (px): half the neighborhood side length. "
            "A radius of R uses a (2R+1) × (2R+1) window.\n\n"
            "Processing uses all available CPU cores via multiprocessing "
            "with overlapping tiles to eliminate seam artefacts."
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
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RADIUS,
                "Median filter radius (px)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=6,
                minValue=1,
                maxValue=50,
            )
        )
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

        # ---- resolve parameters ----------------------------------------
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster_layer is None:
            raise QgsProcessingException("Invalid input raster layer.")

        radius: int = self.parameterAsInt(parameters, self.RADIUS, context)
        output_path: str = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        filter_size = 2 * radius + 1
        feedback.pushInfo(
            f"Median filter: radius={radius} px  →  window={filter_size}×{filter_size}"
        )

        # ---- open with GDAL --------------------------------------------
        src_path = raster_layer.source()
        ds: gdal.Dataset = gdal.Open(src_path, gdal.GA_ReadOnly)
        if ds is None:
            raise QgsProcessingException(f"GDAL could not open: {src_path}")

        band: gdal.Band = ds.GetRasterBand(1)
        cols: int = ds.RasterXSize
        rows: int = ds.RasterYSize
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        nodata = band.GetNoDataValue()

        feedback.pushInfo(f"Raster size: {cols} × {rows} px")

        # ---- read entire array into float32 ----------------------------
        feedback.pushInfo("Reading raster data …")
        data: np.ndarray = band.ReadAsArray().astype(np.float32)
        ds = None  # close source

        nodata_mask: np.ndarray | None = None
        if nodata is not None:
            nodata_mask = np.isnan(data) | (data == nodata)
            # replace nodata with nearest-value padding handled by reflect mode
            data[nodata_mask] = np.nanmedian(data)

        # ---- build tile list -------------------------------------------
        overlap = radius  # overlap = half filter window ensures no seams
        tiles: list[dict] = _build_tiles(rows, cols, TILE_SIZE, overlap)
        n_tiles = len(tiles)
        feedback.pushInfo(
            f"Processing {n_tiles} tiles ({TILE_SIZE}px + {overlap}px overlap) "
            f"across {os.cpu_count()} CPU cores …"
        )

        # ---- parallel processing ---------------------------------------
        tile_args = [(data[t["src_r0"]:t["src_r1"], t["src_c0"]:t["src_c1"]], filter_size)
                     for t in tiles]

        output: np.ndarray = np.empty_like(data)

        # Use spawn context on Windows to avoid QGIS DLL conflicts
        mp_ctx = multiprocessing.get_context("spawn")
        with mp_ctx.Pool(processes=os.cpu_count()) as pool:
            for i, (result, tile) in enumerate(
                zip(pool.imap(_process_tile, tile_args), tiles)
            ):
                if feedback.isCanceled():
                    pool.terminate()
                    raise QgsProcessingException("Cancelled by user.")

                # crop the overlap border from the result before pasting
                r_start = tile["dst_r0"] - tile["src_r0"]
                r_end = r_start + (tile["dst_r1"] - tile["dst_r0"])
                c_start = tile["dst_c0"] - tile["src_c0"]
                c_end = c_start + (tile["dst_c1"] - tile["dst_c0"])
                output[tile["dst_r0"]:tile["dst_r1"], tile["dst_c0"]:tile["dst_c1"]] = (
                    result[r_start:r_end, c_start:c_end]
                )

                feedback.setProgress(int((i + 1) / n_tiles * 100))

        # ---- restore nodata --------------------------------------------
        if nodata_mask is not None:
            fill = float(nodata) if nodata is not None else np.nan
            output[nodata_mask] = fill

        # ---- write GeoTIFF ---------------------------------------------
        feedback.pushInfo(f"Writing output: {output_path}")
        driver: gdal.Driver = gdal.GetDriverByName("GTiff")
        out_ds: gdal.Dataset = driver.Create(
            output_path,
            cols,
            rows,
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                     "BLOCKXSIZE=512", "BLOCKYSIZE=512", "BIGTIFF=IF_SAFER"],
        )
        if out_ds is None:
            raise QgsProcessingException(f"Could not create output file: {output_path}")

        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        out_band: gdal.Band = out_ds.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)
        out_band.WriteArray(output)
        out_band.FlushCache()
        out_ds = None  # close + flush

        feedback.pushInfo("Done.")
        return {self.OUTPUT: output_path}


# ---------------------------------------------------------------------------
# Tiling helper
# ---------------------------------------------------------------------------

def _build_tiles(rows: int, cols: int, tile_size: int, overlap: int) -> list[dict]:
    """
    Build a list of tile descriptors.

    Each tile has:
      src_{r,c}{0,1}  — padded read region (includes overlap)
      dst_{r,c}{0,1}  — write region in the output array (no overlap)
    """
    tiles = []
    n_row = math.ceil(rows / tile_size)
    n_col = math.ceil(cols / tile_size)

    for ri in range(n_row):
        for ci in range(n_col):
            dst_r0 = ri * tile_size
            dst_r1 = min(dst_r0 + tile_size, rows)
            dst_c0 = ci * tile_size
            dst_c1 = min(dst_c0 + tile_size, cols)

            src_r0 = max(dst_r0 - overlap, 0)
            src_r1 = min(dst_r1 + overlap, rows)
            src_c0 = max(dst_c0 - overlap, 0)
            src_c1 = min(dst_c1 + overlap, cols)

            tiles.append({
                "dst_r0": dst_r0, "dst_r1": dst_r1,
                "dst_c0": dst_c0, "dst_c1": dst_c1,
                "src_r0": src_r0, "src_r1": src_r1,
                "src_c0": src_c0, "src_c1": src_c1,
            })

    return tiles
