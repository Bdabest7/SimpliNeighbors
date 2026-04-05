# SimpliNeighbors — Fast DSM Surface Smoothing Plugin for QGIS 4
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later


def classFactory(iface):
    from .simpli_neighbors import SimpliNeighborsPlugin
    return SimpliNeighborsPlugin(iface)
