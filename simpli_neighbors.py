# SimpliNeighbors — Fast DSM Surface Smoothing Plugin for QGIS 4
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later

from qgis.core import QgsApplication
from .simpli_neighbors_provider import SimpliNeighborsProvider


class SimpliNeighborsPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        # qgis_process calls only this; desktop QGIS calls initGui.
        self.provider = SimpliNeighborsProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
