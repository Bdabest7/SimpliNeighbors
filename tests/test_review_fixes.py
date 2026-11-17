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



class Finding7FailedTileTests(unittest.TestCase):
    def _run(self, fake, feedback=None, threads=2):
        data = np.zeros((2048, 2048), dtype=np.float32)
        out = np.empty_like(data)
        tiles = alg._build_tiles(2048, 2048, 256, 2)  # 64 tiles
        with mock.patch.object(alg, '_filter_tile', side_effect=fake):
            alg._filter_tiles(data, out, tiles, 5, alg.MODE_MEDIAN, 1.0, threads,
                              feedback or Feedback())
        return len(tiles)

    def test_failed_tile_does_not_wait_for_the_rest(self):
        import threading
        import time
        calls = []
        lock = threading.Lock()

        def fake(tile, *a):
            with lock:
                calls.append(1)
                n = len(calls)
            if n == 1:
                raise ValueError('tile failed')
            time.sleep(0.3)
            return tile

        t0 = time.time()
        with self.assertRaises(ValueError):
            self._run(fake)
        elapsed = time.time() - t0
        # 64 tiles x 0.3 s on 2 threads would be about 10 s if it waited for all of them.
        self.assertLess(elapsed, 3.0)
        self.assertLess(len(calls), 20)

    def test_cancel_stops_quickly(self):
        import time

        def fake(tile, *a):
            time.sleep(0.2)
            return tile

        t0 = time.time()
        with self.assertRaises(alg.QgsProcessingException):
            self._run(fake, feedback=Feedback(cancel_after=1))
        self.assertLess(time.time() - t0, 3.0)



class Finding9NumbaCacheTests(unittest.TestCase):
    PLUGINS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _run(self, env_extra, remove=()):
        import subprocess
        import sys
        code = (
            "import os, sys\n"
            "assert 'numba' not in sys.modules\n"
            "from SimpliNeighbors import simpli_bilateral_numba as m\n"
            "import numba\n"
            "print(os.environ['NUMBA_CACHE_DIR'])\n"
            "print(numba.config.CACHE_DIR)\n"
            "print(m._cpu_kernel.stats if hasattr(m, '_cpu_kernel') else 'nokernel')\n"
        )
        env = dict(os.environ)
        for k in remove:
            env.pop(k, None)
        env.update(env_extra)
        env['PYTHONPATH'] = self.PLUGINS
        r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                           text=True, cwd=tempfile.gettempdir(), timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        return r.stdout.strip().splitlines()

    def test_default_is_per_account_under_localappdata_and_set_before_numba(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run({'LOCALAPPDATA': d}, remove=('NUMBA_CACHE_DIR',))
            want = os.path.join(d, 'SimpliNeighbors', 'numba-cache')
            self.assertEqual(os.path.normcase(out[0]), os.path.normcase(want))
            self.assertEqual(os.path.normcase(out[1]), os.path.normcase(want))
            self.assertTrue(os.path.isdir(want))

    def test_explicit_cache_dir_is_respected_and_plugin_folder_is_not_used(self):
        with tempfile.TemporaryDirectory() as d:
            custom = os.path.join(d, 'mine')
            out = self._run({'NUMBA_CACHE_DIR': custom})
            self.assertEqual(os.path.normcase(out[1]), os.path.normcase(custom))
            plugin_dir = os.path.join(self.PLUGINS, 'SimpliNeighbors')
            self.assertFalse(os.path.normcase(out[1]).startswith(os.path.normcase(plugin_dir)))



class Finding12CudaPathTests(unittest.TestCase):
    PLUGINS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def setUp(self):
        from .. import simpli_bilateral_numba as m
        self.m = m
        m.CUDA_NOTES.clear()

    def test_version_sorting_is_numeric(self):
        names = ['v9.0', 'v12.8', 'v13.2', 'v10.1']
        self.assertEqual(sorted(names, key=self.m._version_key, reverse=True)[:2], ['v13.2', 'v12.8'])
        self.assertEqual(sorted(names, reverse=True)[0], 'v9.0')  # what a string sort did

    def test_cuda_path_outside_program_files_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'nvvm'))
            with mock.patch.dict(os.environ, {'CUDA_PATH': d}):
                root = self.m._find_cuda_root()
            self.assertNotEqual(os.path.normcase(root), os.path.normcase(d))
            self.assertTrue(any('not under Program Files' in n for n in self.m.CUDA_NOTES))

    def test_fake_program_files_env_cannot_add_a_root(self):
        with tempfile.TemporaryDirectory() as d:
            evil = os.path.join(d, 'Evil')
            os.makedirs(os.path.join(evil, 'nvvm'))
            with mock.patch.dict(os.environ, {'ProgramFiles': evil, 'ProgramW6432': evil, 'CUDA_PATH': evil}):
                self.assertFalse(self.m._is_trusted_cuda_dir(evil))
                self.assertNotEqual(os.path.normcase(self.m._find_cuda_root()), os.path.normcase(evil))

    def test_trusted_program_files_path(self):
        self.assertTrue(self.m._is_trusted_cuda_dir(r'C:\Program Files\NVIDIA GPU Computing Toolkit'))
        self.assertFalse(self.m._is_trusted_cuda_dir(r'C:\Program Files (x86)\x'))
        self.assertFalse(self.m._is_trusted_cuda_dir(r'C:\Users'))

    def test_cuda_off_switch_in_fresh_interpreter(self):
        import subprocess
        import sys
        code = ("import os\n"
                "from SimpliNeighbors import simpli_bilateral_numba as m\n"
                "print(m.CUDA_DISABLED, m._CUDA_OK, repr(m.CUDA_ROOT), m._NUMBA_OK)\n")
        env = dict(os.environ, SIMPLINEIGHBORS_CUDA='0', PYTHONPATH=self.PLUGINS)
        r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                           cwd=tempfile.gettempdir(), timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        self.assertEqual(r.stdout.split()[:3], ['True', 'False', "''"])

if __name__ == '__main__':
    unittest.main()
