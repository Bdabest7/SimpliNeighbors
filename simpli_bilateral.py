# SimpliNeighbors — bundled bilateral filter (no OpenCV required)
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Pure NumPy implementation — NumPy's C-level array operations provide
# comparable throughput to OpenCV for the tile sizes we use (512×512 px).
#
# Algorithm
# ---------
# The bilateral filter replaces each pixel with a weighted average of its
# neighbourhood.  Each neighbour receives two independent Gaussian weights:
#
#   w_spatial(p, q) = exp( -||p - q||² / (2 · σ_space²) )
#       Distance from the centre pixel — suppresses far-away neighbours.
#
#   w_range(p, q)   = exp( -(I(p) - I(q))² / (2 · σ_color²) )
#       Elevation difference — suppresses neighbours that differ strongly
#       in value, i.e. the filter stops at topographic edges.
#
# The combined weight is the product of the two, so:
#   • Within a smooth slope  → all weights stay high → strong smoothing
#   • Across a sharp edge    → range weight collapses → edge is preserved
#
# Parameters
# ----------
# filter_size  : int   — side length of the neighbourhood window (odd)
# sigma_space  : float — spatial Gaussian σ in pixels (try filter_size / 3)
# sigma_color  : float — intensity Gaussian σ in elevation units
#                        (e.g. 0.05 – 0.5 for metre-scale DTMs)

from __future__ import annotations
import numpy as np


def bilateral_filter(
    data: np.ndarray,
    filter_size: int,
    sigma_space: float,
    sigma_color: float,
) -> np.ndarray:
    """
    Apply an edge-preserving bilateral filter to a 2-D float32 raster tile.

    Parameters
    ----------
    data        : 2-D float32 numpy array (single raster band tile)
    filter_size : neighbourhood window side in pixels (odd integer)
    sigma_space : spatial Gaussian sigma (pixels)
    sigma_color : range/intensity Gaussian sigma (same units as elevation data)

    Returns
    -------
    Filtered float32 array, same shape as *data*.
    """
    data = data.astype(np.float32)
    radius = filter_size // 2
    h, w = data.shape

    # Reflect-pad so border pixels get a full neighbourhood
    padded = np.pad(data, radius, mode="reflect")

    # Pre-compute the spatial Gaussian kernel once (shape: filter_size × filter_size)
    ky, kx = np.mgrid[0:filter_size, 0:filter_size]
    cy, cx = radius, radius
    spatial_kernel = np.exp(
        -((kx - cx) ** 2 + (ky - cy) ** 2) / (2.0 * float(sigma_space) ** 2)
    ).astype(np.float32)

    # Cache reciprocal to avoid per-iteration division
    inv_2_sc2 = np.float32(1.0 / (2.0 * float(sigma_color) ** 2))

    output   = np.zeros((h, w), dtype=np.float32)
    wsum     = np.zeros((h, w), dtype=np.float32)

    # Iterate over each offset in the window.
    # Each iteration is one NumPy vectorised operation over the full tile —
    # no Python-level pixel loop.
    for dy in range(filter_size):
        for dx in range(filter_size):
            neighbour = padded[dy : dy + h, dx : dx + w]  # (h, w)

            # Range weight: how similar is this neighbour in elevation?
            diff  = data - neighbour
            r_w   = np.exp(-(diff * diff) * inv_2_sc2)       # (h, w)

            # Combined weight
            total_w = np.float32(spatial_kernel[dy, dx]) * r_w  # (h, w)

            output += total_w * neighbour
            wsum   += total_w

    # Normalise (wsum is always > 0 because the centre pixel has weight 1)
    return output / wsum
