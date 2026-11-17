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



class Finding8BackendTests(unittest.TestCase):
    def setUp(self):
        from .. import simpli_bilateral as sb
        self.sb = sb

    def test_auto_and_pins(self):
        self.assertIn(self.sb.resolve_backend(None), ('CUDA', 'Numba CPU', 'NumPy'))
        self.assertEqual(self.sb.resolve_backend('numpy'), 'NumPy')
        self.assertEqual(self.sb.resolve_backend('auto'), self.sb.get_bilateral_backend())

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(self.sb.BackendUnavailable):
            self.sb.resolve_backend('opencl')

    def test_missing_cuda_fails_instead_of_falling_back(self):
        from .. import simpli_bilateral_numba as nb
        with mock.patch.object(nb, '_CUDA_OK', False):
            with self.assertRaises(self.sb.BackendUnavailable):
                self.sb.resolve_backend('cuda')

    def test_cuda_switched_off_fails_when_pinned(self):
        from .. import simpli_bilateral_numba as nb
        with mock.patch.object(nb, 'CUDA_DISABLED', True):
            with self.assertRaises(self.sb.BackendUnavailable):
                self.sb.resolve_backend('cuda')

    def test_missing_numba_fails_instead_of_falling_back(self):
        from .. import simpli_bilateral_numba as nb
        with mock.patch.object(nb, '_NUMBA_OK', False):
            with self.assertRaises(self.sb.BackendUnavailable):
                self.sb.resolve_backend('numba')

    def test_kernel_is_nogil(self):
        from .. import simpli_bilateral_numba as nb
        if not nb._NUMBA_OK:
            self.skipTest('numba not installed')
        self.assertTrue(nb._cpu_kernel.targetoptions.get('nogil'))

    def test_pinned_backends_agree(self):
        rng = np.random.default_rng(7)
        data = (rng.random((64, 64)) * 0.2).astype(np.float32)
        a = self.sb.bilateral_filter(data, 7, 2.0, 0.1, backend='NumPy')
        from .. import simpli_bilateral_numba as nb
        if nb._NUMBA_OK:
            b = self.sb.bilateral_filter(data, 7, 2.0, 0.1, backend='Numba CPU')
            self.assertLess(float(np.abs(a - b).max()), 1e-3)

    def test_parameter_exists_and_is_advanced(self):
        a = alg.SimpliNeighborsAlgorithm()
        a.initAlgorithm()
        p = a.parameterDefinition('BACKEND')
        self.assertIsNotNone(p)
        self.assertTrue(p.flags() & alg.QgsProcessingParameterDefinition.Flag.Advanced)
        self.assertEqual(p.defaultValue(), 0)



class Finding13ValidationTests(unittest.TestCase):
    def test_resolution_parser_rejects_garbage(self):
        self.assertIsNone(alg._parse_resolution('default'))
        self.assertIsNone(alg._parse_resolution(' Default '))
        self.assertIsNone(alg._parse_resolution(''))
        self.assertEqual(alg._parse_resolution('0.25'), 0.25)
        for bad in ('abc', '-1', '0', 'nan', 'inf', '1,5', '0.1m'):
            with self.assertRaises(ValueError, msg=bad):
                alg._parse_resolution(bad)

    def test_threads_maximum_is_not_this_machines_core_count(self):
        a = alg.SimpliNeighborsAlgorithm()
        a.initAlgorithm()
        p = a.parameterDefinition('THREADS')
        self.assertEqual(p.maximum(), alg.MAX_THREADS)
        self.assertGreaterEqual(alg.MAX_THREADS, 256)
        self.assertEqual(p.minimum(), 1)
        self.assertEqual(p.defaultValue(), alg._CPU_COUNT)  # default unchanged

    def test_tile_cap(self):
        self.assertEqual(alg._max_tiles(), alg.DEFAULT_MAX_TILES)
        self.assertGreater(alg.DEFAULT_MAX_TILES, 10 * 1200)  # Jackson CC DTM is about 1,200 tiles
        with mock.patch.dict(os.environ, {'SIMPLINEIGHBORS_MAX_TILES': '12'}):
            self.assertEqual(alg._max_tiles(), 12)
        for bad in ('lots', '0', '-3'):
            with mock.patch.dict(os.environ, {'SIMPLINEIGHBORS_MAX_TILES': bad}):
                with self.assertRaises(alg.QgsProcessingException):
                    alg._max_tiles()



class Finding14DepsTests(unittest.TestCase):
    def test_check_reports_versions_and_pins(self):
        from .. import simpli_deps
        lines, missing = simpli_deps.check_dependencies()
        self.assertEqual(missing, [])
        self.assertTrue(any(l.startswith('dep scipy') for l in lines))
        pins = simpli_deps._pins()
        for name in ('numpy', 'scipy', 'numba', 'llvmlite'):
            self.assertIn(name, pins)

    def test_missing_scipy_is_named(self):
        from .. import simpli_deps
        with mock.patch.dict('sys.modules', {'scipy': None}):
            lines, missing = simpli_deps.check_dependencies()
            self.assertIn('scipy', missing)
            with self.assertRaises(ImportError) as cm:
                simpli_deps.require_for(None, mode_needs_scipy=True)
            self.assertIn('scipy==', str(cm.exception))
            simpli_deps.require_for(None, mode_needs_scipy=False)  # bilateral does not need scipy

    def test_numba_required_only_when_asked(self):
        from .. import simpli_deps
        with mock.patch.dict('sys.modules', {'numba': None}):
            _, missing = simpli_deps.check_dependencies()
            self.assertNotIn('numba', missing)
            _, missing = simpli_deps.check_dependencies(['numba'])
            self.assertIn('numba', missing)

    def test_cli_exit_codes(self):
        import subprocess
        import sys
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'simpli_deps.py')
        r = subprocess.run([sys.executable, path, '--require', 'numba'], capture_output=True, text=True,
                           cwd=tempfile.gettempdir(), timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr[-400:])
        self.assertIn('dep numba', r.stdout)



class Finding15GdalSettingsTests(unittest.TestCase):
    def tearDown(self):
        gdal.UseExceptions()  # this test module's own default

    def test_exceptions_state_restored_after_run_and_failure(self):
        a = alg.SimpliNeighborsAlgorithm()
        seen = []

        def ok(*args):
            seen.append(gdal.GetUseExceptions())
            return {}

        def boom(*args):
            seen.append(gdal.GetUseExceptions())
            raise RuntimeError('x')

        gdal.DontUseExceptions()
        with mock.patch.object(a, '_process', side_effect=ok):
            a.processAlgorithm({}, None, None)
        self.assertEqual(seen, [True])          # on during the run
        self.assertFalse(gdal.GetUseExceptions())  # restored after

        with mock.patch.object(a, '_process', side_effect=boom):
            with self.assertRaises(RuntimeError):
                a.processAlgorithm({}, None, None)
        self.assertFalse(gdal.GetUseExceptions())

        gdal.UseExceptions()
        with mock.patch.object(a, '_process', side_effect=ok):
            a.processAlgorithm({}, None, None)
        self.assertTrue(gdal.GetUseExceptions())  # was on before, stays on

    def test_importing_the_module_does_not_change_the_setting(self):
        import subprocess
        import sys
        code = ("from osgeo import gdal\n"
                "gdal.DontUseExceptions()\n"
                "from SimpliNeighbors import simpli_neighbors_algorithm\n"
                "print(gdal.GetUseExceptions())\n")
        plugins = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        r = subprocess.run([sys.executable, '-c', code], env=dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [plugins, os.environ.get('PYTHONPATH', '')]))),
                           capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=300)
        self.assertIn(r.stdout.strip().splitlines()[-1], ('0', 'False'), r.stderr[-400:])



class Finding4AtomicOutputTests(unittest.TestCase):
    GT = (0.0, 0.5, 0.0, 10.0, 0.0, -0.5)

    def _make(self, d):
        return alg.AtomicGeoTiff(os.path.join(d, 'out.tif'), 64, 64, self.GT, '', -32767)

    def test_success_replaces_output_and_leaves_no_partial(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._make(d)
            self.assertFalse(os.path.exists(os.path.join(d, 'out.tif')))  # nothing there while writing
            out.write(np.full((64, 64), 5, dtype=np.float32))
            self.assertFalse(os.path.exists(os.path.join(d, 'out.tif')))
            out.commit()
            self.assertTrue(os.path.exists(os.path.join(d, 'out.tif')))
            self.assertFalse(os.path.exists(os.path.join(d, 'out.tif.partial')))
            ds = gdal.Open(os.path.join(d, 'out.tif'))
            self.assertEqual(float(ds.GetRasterBand(1).ReadAsArray()[3, 3]), 5.0)
            self.assertEqual(ds.GetRasterBand(1).GetNoDataValue(), -32767)
            ds = None

    def test_failure_leaves_no_output_and_keeps_old_one(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.path.join(d, 'out.tif')
            open(old, 'wb').write(b'old complete result')
            out = self._make(d)
            out.write(np.zeros((32, 64), dtype=np.float32), 0)   # half the rows only
            out.abandon()                                         # run failed here
            self.assertEqual(open(old, 'rb').read(), b'old complete result')
            self.assertTrue(os.path.exists(old + '.partial'))     # kept, clearly incomplete

    def test_fresh_failure_leaves_nothing_at_output_path(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._make(d)
            out.write(np.zeros((32, 64), dtype=np.float32), 0)
            out.abandon()
            self.assertFalse(os.path.exists(os.path.join(d, 'out.tif')))

    def test_algorithm_wraps_write_in_atomic_writer(self):
        with open(alg.__file__, encoding='utf-8') as f:
            src = f.read()
        self.assertIn('AtomicGeoTiff(output_path', src)
        self.assertNotIn('driver.Create(\n            output_path', src)

if __name__ == '__main__':
    unittest.main()
