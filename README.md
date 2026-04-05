# SimpliNeighbors

A fast, crash-free QGIS 4.0 plugin for DSM surface smoothing via median filter — a drop-in replacement for the built-in GRASS `r.neighbors` plugin.

## Overview

SimpliNeighbors applies a parallel tiled median filter to any single-band raster (DSMs, DTMs, etc.). It is functionally equivalent to running:

```
r.neighbors input=<raster> output=<result> method=median size=<2*radius+1>
```

but implemented entirely in Python/NumPy/SciPy — no GRASS installation required, no crashes.

### Why not just use the built-in plugin?

The QGIS GRASS `r.neighbors` plugin can crash due to GRASS environment setup failures, incompatible GRASS versions, or memory issues on large rasters. SimpliNeighbors bypasses GRASS entirely and processes rasters in-memory using `scipy.ndimage.median_filter` with multiprocessing tile parallelism.

## UI

The interface replicates the Pix4D Survey DSM surface smoothing panel:

```
Surface smoothing
[ Low ══════●═══════════ High ]  [ 6 ]
       Median filter radius (px)
```

- **Radius 1–50 px** — controls the filter half-window. Radius `R` uses a `(2R+1) × (2R+1)` neighborhood (e.g. radius 6 → 13×13 window).
- A **Low/High slider** and a **spinbox** are bidirectionally synced.

## Performance

- Reads the entire raster into a `float32` NumPy array.
- Splits it into **512×512 px tiles** with `radius`-pixel overlap to prevent seam artefacts.
- Distributes all tiles across **all CPU cores** via `multiprocessing.Pool`.
- Writes a **LZW-compressed, tiled GeoTIFF** for fast I/O.

On a modern workstation, a 10,000×10,000 DSM with radius=6 completes in a few seconds.

## Requirements

| Dependency | Version |
|---|---|
| QGIS | 4.0.0+ |
| Python | 3.9+ (bundled with QGIS) |
| NumPy | 1.20+ (bundled with QGIS) |
| SciPy | 1.7+ |
| GDAL/OGR | 3.x (bundled with QGIS) |

> **SciPy** is the only dependency that may not be pre-installed. Install it inside the QGIS Python environment:
>
> **Windows (OSGeo4W shell):**
> ```
> python -m pip install scipy
> ```
>
> **Linux/macOS:**
> ```
> pip3 install scipy
> ```

## Installation

1. Download or clone this repository.
2. Copy the `SimpliNeighbors` folder into your QGIS plugins directory:
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
3. Open QGIS → **Plugins → Manage and Install Plugins → Installed** → enable **SimpliNeighbors**.
4. The algorithm appears in **Processing Toolbox → SimpliNeighbors → Surface Smoothing (Median Filter)**.

## Usage

1. Open **Processing Toolbox** (`Ctrl+Alt+T`).
2. Navigate to **SimpliNeighbors → Surface Smoothing (Median Filter)**.
3. Set **Input raster** to your DSM layer.
4. Adjust the **Median filter radius (px)** slider or spinbox (default: 6).
5. Set an output path.
6. Click **Run**. Progress is reported per tile; cancellation is supported.

## Parameters

| Parameter | Description | Range | Default |
|---|---|---|---|
| Input raster | Single-band raster (DSM/DTM/etc.) | — | — |
| Median filter radius (px) | Half the filter window size. Window = `(2R+1)²` | 1–50 | 6 |
| Smoothed output | Output GeoTIFF path | — | — |

## Algorithm Details

The median filter replaces each pixel with the **median** of its neighborhood, which:
- Removes spike noise and outliers (buildings, vegetation artefacts in photogrammetric DSMs).
- Preserves sharp edges better than mean/Gaussian smoothing.
- Matches the behaviour of `r.neighbors method=median`.

Nodata values are temporarily filled with the raster median before filtering and restored afterward, preventing filter contamination across data boundaries.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
