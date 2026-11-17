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




class Finding3ResampleTests(unittest.TestCase):
    def test_memory_warp_equals_file_warp(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, 'src.tif')
            drv = gdal.GetDriverByName('GTiff')
            ds = drv.Create(src, 300, 200, 1, gdal.GDT_Float32)
            ds.SetGeoTransform((0, 0.05, 0, 10, 0, -0.05))
            ds.GetRasterBand(1).SetNoDataValue(-32767)
            ds.GetRasterBand(1).WriteArray(np.random.default_rng(3).random((200, 300)).astype(np.float32))
            ds = None
            old = os.path.join(d, 'old.tif')
            w = gdal.Warp(old, src, options=gdal.WarpOptions(
                xRes=0.08, yRes=0.08, resampleAlg=gdal.GRA_Bilinear,
                creationOptions=['COMPRESS=LZW', 'TILED=YES']))
            mem = gdal.Warp('', src, options=gdal.WarpOptions(
                format='MEM', xRes=0.08, yRes=0.08, resampleAlg=gdal.GRA_Bilinear))
            w = None  # flush and close
            od = gdal.Open(old)
            a = od.GetRasterBand(1).ReadAsArray()
            od = None
            b = mem.GetRasterBand(1).ReadAsArray()
            self.assertTrue(np.array_equal(a, b))

    def test_algorithm_source_has_no_fixed_temp_path(self):
        with open(alg.__file__, encoding='utf-8') as f:
            src = f.read()
        self.assertNotIn('simplineighbors_resampled', src)

if __name__ == '__main__':
    unittest.main()
