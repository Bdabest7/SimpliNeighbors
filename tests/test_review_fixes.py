r"""Tests for the 2026-10-13 review fixes (run with QGIS's Python, from the plugins folder):

    cd <profile>\python\plugins
    python -m unittest SimpliNeighbors.tests.test_review_fixes -v

Separate from ../test_plugin.py, which is a flat-import script (three of its
tests fail at 5893055 with 'attempted relative import', unrelated to these fixes).
"""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from osgeo import gdal

from .. import simpli_neighbors_algorithm as alg

gdal.UseExceptions()


class Feedback:
    def __init__(self, cancel_after=None):
        self.info, self.warn, self.progress = [], [], []
        self.cancel_after = cancel_after

    def pushInfo(self, m): self.info.append(m)
    def pushWarning(self, m): self.warn.append(m)
    def pushDebugInfo(self, m): self.info.append(m)
    def reportError(self, m): self.warn.append(m)
    def setProgress(self, p): self.progress.append(p)
    def isCanceled(self):
        return self.cancel_after is not None and len(self.progress) >= self.cancel_after


class Finding2PresetTests(unittest.TestCase):
    A = alg.SimpliNeighborsAlgorithm

    def test_default_without_anything_is_preset(self):
        self.assertEqual(alg.resolve_settings({}, 1), (True, None))

    def test_preset_passed_wins_as_before(self):
        p = {self.A.PRESET: 1, self.A.RADIUS: 5, self.A.SMOOTH_MODE: 0, self.A.GAUSSIAN_SIGMA: 1.5}
        self.assertEqual(alg.resolve_settings(p, 1), (True, None))

    def test_custom_preset_passed(self):
        self.assertEqual(alg.resolve_settings({self.A.PRESET: 0, self.A.RADIUS: 5}, 0), (False, None))

    def test_explicit_without_preset_wins(self):
        use_preset, note = alg.resolve_settings({self.A.RADIUS: 5}, 1)
        self.assertFalse(use_preset)
        self.assertIn('RADIUS', note)

    def test_brentons_parameters_unchanged(self):
        p = {self.A.PRESET: 1, self.A.RADIUS: 16, self.A.SMOOTH_MODE: 2, self.A.GAUSSIAN_SIGMA: 0.1}
        self.assertEqual(alg.resolve_settings(p, 1), (True, None))


if __name__ == '__main__':
    unittest.main()
