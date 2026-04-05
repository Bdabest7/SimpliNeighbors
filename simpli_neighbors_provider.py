# SimpliNeighbors — Fast DSM Surface Smoothing Plugin for QGIS 4
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later

from qgis.core import QgsProcessingProvider
from .simpli_neighbors_algorithm import SimpliNeighborsAlgorithm


class SimpliNeighborsProvider(QgsProcessingProvider):

    def loadAlgorithms(self):
        self.addAlgorithm(SimpliNeighborsAlgorithm())

    def id(self):
        return "simplineighbors"

    def name(self):
        return "SimpliNeighbors"

    def longName(self):
        return "SimpliNeighbors — Fast Surface Smoothing"
