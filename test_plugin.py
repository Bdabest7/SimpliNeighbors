"""
SimpliNeighbors — plugin smoke tests.

Run from the OSGeo4W shell (which has QGIS on PYTHONPATH):

    cd %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\SimpliNeighbors
    python test_plugin.py

Tests cover the parts most likely to fail on QGIS version changes:
  1. All imports resolve
  2. Algorithm instantiates and lists expected parameters
  3. Wrapper constructors accept (param, MockDialog, row, col) — the QGIS 4.x calling convention
  4. Resolution string parser handles all expected inputs
  5. Tiling helper produces correct tile counts and overlap
"""

import sys
import types
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []


def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {e}")
        traceback.print_exc()
        results.append((name, False, e))


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

print("\n--- 1. Imports ---")

def test_import_algorithm():
    from simpli_neighbors_algorithm import SimpliNeighborsAlgorithm  # noqa: F401

def test_import_provider():
    from simpli_neighbors_provider import SimpliNeighborsProvider  # noqa: F401

def test_import_plugin():
    from simpli_neighbors import SimpliNeighborsPlugin  # noqa: F401

test("import simpli_neighbors_algorithm", test_import_algorithm)
test("import simpli_neighbors_provider",  test_import_provider)
test("import simpli_neighbors (plugin)",  test_import_plugin)

# ---------------------------------------------------------------------------
# 2. Algorithm parameters
# ---------------------------------------------------------------------------

print("\n--- 2. Algorithm parameters ---")

def test_algorithm_params():
    from simpli_neighbors_algorithm import SimpliNeighborsAlgorithm
    alg = SimpliNeighborsAlgorithm()
    alg.initAlgorithm()
    names = [p.name() for p in alg.parameterDefinitions()]
    for expected in ("INPUT", "PRESET", "SMOOTH_MODE", "RADIUS", "GAUSSIAN_SIGMA",
                     "THREADS", "RESOLUTION", "OUTPUT"):
        assert expected in names, f"Missing parameter: {expected}"

def test_presets_valid():
    from simpli_neighbors_algorithm import (
        PRESETS, PRESET_CUSTOM, MODE_TWO_PASS, MODE_MEDIAN, MODE_BILATERAL
    )
    valid_modes = {MODE_TWO_PASS, MODE_MEDIAN, MODE_BILATERAL}
    for i, (label, mode, radius, sigma) in enumerate(PRESETS):
        if i == PRESET_CUSTOM:
            assert mode is None
            continue
        assert mode in valid_modes,          f"Preset {i} has invalid mode {mode}"
        assert isinstance(radius, int),      f"Preset {i} radius must be int"
        assert 1 <= radius <= 50,            f"Preset {i} radius {radius} out of range"
        assert isinstance(sigma, float),     f"Preset {i} sigma must be float"
        assert sigma > 0,                    f"Preset {i} sigma must be > 0"

def test_algorithm_identity():
    from simpli_neighbors_algorithm import SimpliNeighborsAlgorithm
    alg = SimpliNeighborsAlgorithm()
    assert alg.name() == "surfacesmoothing"
    assert alg.groupId() == "raster"
    inst = alg.createInstance()
    assert isinstance(inst, SimpliNeighborsAlgorithm)

test("algorithm has all 8 expected parameters", test_algorithm_params)
test("algorithm name/group/createInstance",     test_algorithm_identity)
test("all presets have valid mode/radius/sigma", test_presets_valid)

# ---------------------------------------------------------------------------
# 3. Wrapper constructors — QGIS 4.x calling convention
# ---------------------------------------------------------------------------

print("\n--- 3. Wrapper constructors (QGIS 4.x convention) ---")

# Minimal mock objects so we can test wrappers without a running QGIS instance
try:
    from qgis.core import QgsProcessingParameterNumber, QgsProcessingParameterString
    from qgis.gui import QgsProcessingGui

    class _MockAlgorithmDialog:
        """Simulates what QGIS 4.x passes as 'dialog' arg."""
        __class__ = type("AlgorithmDialog", (), {})()

    class _MockBatchDialog:
        __class__ = type("BatchAlgorithmDialog", (), {})()

    # Patch __class__.__name__ properly
    class AlgorithmDialog:
        pass

    class BatchAlgorithmDialog:
        pass

    mock_std   = AlgorithmDialog()
    mock_batch = BatchAlgorithmDialog()

    radius_param  = QgsProcessingParameterNumber("RADIUS", "r", defaultValue=6)
    threads_param = QgsProcessingParameterNumber("THREADS", "t", defaultValue=4)
    res_param     = QgsProcessingParameterString("RESOLUTION", "res", defaultValue="default")

    def test_radius_wrapper():
        from simpli_neighbors_widget import SimpliNeighborsRadiusWrapper
        w = SimpliNeighborsRadiusWrapper(radius_param, mock_std, 0, 0)
        assert w.dialogType == QgsProcessingGui.WidgetType.Standard

    def test_cores_wrapper():
        from simpli_neighbors_widget import SimpliNeighborsCoresWrapper
        w = SimpliNeighborsCoresWrapper(threads_param, mock_std, 0, 0)
        assert w.dialogType == QgsProcessingGui.WidgetType.Standard

    def test_resolution_wrapper():
        from simpli_neighbors_widget import SimpliNeighborsResolutionWrapper
        w = SimpliNeighborsResolutionWrapper(res_param, mock_std, 0, 0)
        assert w.dialogType == QgsProcessingGui.WidgetType.Standard

    def test_batch_dialog_type():
        from simpli_neighbors_widget import SimpliNeighborsCoresWrapper
        w = SimpliNeighborsCoresWrapper(threads_param, mock_batch, 0, 0)
        assert w.dialogType == QgsProcessingGui.WidgetType.Batch

    test("RadiusWrapper(param, AlgorithmDialog, row, col)",       test_radius_wrapper)
    test("CoresWrapper(param, AlgorithmDialog, row, col)",        test_cores_wrapper)
    test("ResolutionWrapper(param, AlgorithmDialog, row, col)",   test_resolution_wrapper)
    test("CoresWrapper resolves BatchAlgorithmDialog → Batch",    test_batch_dialog_type)

except ImportError as e:
    print(f"  SKIP  (QGIS not on PYTHONPATH: {e})")

# ---------------------------------------------------------------------------
# 3b. Bundled bilateral filter
# ---------------------------------------------------------------------------

print("\n--- 3b. Bundled bilateral filter ---")

def test_bilateral_import():
    from simpli_bilateral import bilateral_filter  # noqa: F401

def test_bilateral_edge_preservation():
    """Bilateral filter must preserve a sharp edge better than Gaussian."""
    import numpy as np
    from simpli_bilateral import bilateral_filter
    from scipy.ndimage import gaussian_filter

    # Sharp step: left half = 0, right half = 1 (simulates bunker lip)
    data = np.zeros((64, 64), dtype=np.float32)
    data[:, 32:] = 1.0

    gauss_result    = gaussian_filter(data, sigma=3.0)
    bilateral_result = bilateral_filter(data, filter_size=13,
                                        sigma_space=4.0, sigma_color=0.1)

    # Measure edge sharpness: std of a vertical slice through the edge
    gauss_edge     = gauss_result[:, 28:36].std()
    bilateral_edge = bilateral_result[:, 28:36].std()

    assert bilateral_edge > gauss_edge, (
        f"Bilateral edge std ({bilateral_edge:.4f}) should be sharper "
        f"than Gaussian ({gauss_edge:.4f})"
    )

def test_bilateral_slope_smoothing():
    """Bilateral must smooth a continuous slope (no sharp edges)."""
    import numpy as np
    from simpli_bilateral import bilateral_filter

    # Gentle ramp with noise — simulates fairway slope
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    noisy = ramp + np.random.default_rng(42).normal(0, 0.02, ramp.shape).astype(np.float32)
    result = bilateral_filter(noisy, filter_size=9, sigma_space=3.0, sigma_color=0.5)

    # Result must be smoother than input (lower gradient variance)
    assert result.std() < noisy.std(), "Bilateral must reduce noise on a smooth slope"

test("bilateral filter module imports",                      test_bilateral_import)
test("bilateral preserves sharp edges (bunker lip)",         test_bilateral_edge_preservation)
test("bilateral smooths gentle slopes (fairway noise)",      test_bilateral_slope_smoothing)

# ---------------------------------------------------------------------------
# 4. Resolution string parser
# ---------------------------------------------------------------------------

print("\n--- 4. Resolution parser ---")

def test_resolution_parser():
    from simpli_neighbors_algorithm import _parse_resolution
    assert _parse_resolution("default") is None
    assert _parse_resolution("Default") is None
    assert _parse_resolution("")        is None
    assert _parse_resolution("0.004")  == 0.004
    assert _parse_resolution("1.0")    == 1.0
    assert _parse_resolution("-1")     is None   # negative → ignored
    assert _parse_resolution("abc")    is None   # garbage → ignored

test("_parse_resolution handles all cases", test_resolution_parser)

# ---------------------------------------------------------------------------
# 5. Tiling helper
# ---------------------------------------------------------------------------

print("\n--- 5. Tiling helper ---")

def test_tile_count():
    import math
    from simpli_neighbors_algorithm import _build_tiles
    rows, cols, size, overlap = 1000, 1000, 512, 6
    tiles = _build_tiles(rows, cols, size, overlap)
    expected = math.ceil(rows / size) * math.ceil(cols / size)
    assert len(tiles) == expected, f"Expected {expected} tiles, got {len(tiles)}"

def test_tile_coverage():
    """Every output pixel must be covered by exactly one dst tile."""
    from simpli_neighbors_algorithm import _build_tiles
    import numpy as np
    rows, cols, size, overlap = 300, 400, 128, 10
    coverage = np.zeros((rows, cols), dtype=np.int32)
    for t in _build_tiles(rows, cols, size, overlap):
        coverage[t["dst_r0"]:t["dst_r1"], t["dst_c0"]:t["dst_c1"]] += 1
    assert coverage.min() == 1, "Some pixels not covered"
    assert coverage.max() == 1, "Some pixels covered more than once"

def test_tile_overlap():
    """src region must extend by at least 'overlap' beyond dst on all sides (where possible)."""
    from simpli_neighbors_algorithm import _build_tiles
    overlap = 8
    for t in _build_tiles(200, 200, 64, overlap):
        if t["dst_c0"] > 0:
            assert t["src_c0"] <= t["dst_c0"] - overlap
        if t["dst_r0"] > 0:
            assert t["src_r0"] <= t["dst_r0"] - overlap

def test_two_pass_output():
    """Two-pass result must differ from median-only on a ramp (no terracing)."""
    import numpy as np
    from simpli_neighbors_algorithm import _filter_tile, MODE_TWO_PASS, MODE_MEDIAN
    # Create a gentle linear ramp — pure median will terrace it
    ramp = np.tile(np.linspace(0, 10, 64, dtype=np.float32), (64, 1))
    result_median   = _filter_tile(ramp, filter_size=13, mode=MODE_MEDIAN,    gaussian_sigma=1.5)
    result_two_pass = _filter_tile(ramp, filter_size=13, mode=MODE_TWO_PASS,  gaussian_sigma=1.5)
    # Two-pass should be smoother: lower std-dev of the gradient
    grad_median    = np.abs(np.diff(result_median,   axis=1)).std()
    grad_two_pass  = np.abs(np.diff(result_two_pass, axis=1)).std()
    assert grad_two_pass < grad_median, (
        f"Two-pass gradient std ({grad_two_pass:.4f}) should be "
        f"less than median-only ({grad_median:.4f})"
    )

def test_filter_tile_modes():
    """All three modes must return float32 arrays of the same shape as input."""
    import numpy as np
    from simpli_neighbors_algorithm import _filter_tile, MODE_TWO_PASS, MODE_MEDIAN, MODE_BILATERAL
    data = np.random.rand(64, 64).astype(np.float32) * 100
    for mode in (MODE_TWO_PASS, MODE_MEDIAN, MODE_BILATERAL):
        result = _filter_tile(data, filter_size=7, mode=mode, gaussian_sigma=1.5)
        assert result.shape == data.shape, f"Mode {mode}: shape mismatch"
        assert result.dtype == np.float32, f"Mode {mode}: not float32"

test("tile count matches ceil(rows/size) * ceil(cols/size)", test_tile_count)
test("every pixel covered exactly once",                     test_tile_coverage)
test("src tiles extend by overlap beyond dst",               test_tile_overlap)
test("two-pass is smoother than median-only on a ramp",      test_two_pass_output)
test("all three filter modes return correct shape/dtype",    test_filter_tile_modes)

# ---------------------------------------------------------------------------
# 6. PyQt6 enum usage in widget file
# ---------------------------------------------------------------------------

print("\n--- 6. PyQt6 enum references ---")

def test_no_pyqt5_enums():
    """Scan widget source for PyQt5-style flat enum access that breaks on PyQt6/QGIS4."""
    import re
    pyqt5_patterns = [
        (r"Qt\.AlignRight\b",   "Qt.AlignRight → Qt.AlignmentFlag.AlignRight"),
        (r"Qt\.AlignLeft\b",    "Qt.AlignLeft → Qt.AlignmentFlag.AlignLeft"),
        (r"Qt\.AlignCenter\b",  "Qt.AlignCenter → Qt.AlignmentFlag.AlignCenter"),
        (r"Qt\.AlignVCenter\b", "Qt.AlignVCenter → Qt.AlignmentFlag.AlignVCenter"),
        (r"Qt\.Horizontal\b",   "Qt.Horizontal → Qt.Orientation.Horizontal"),
        (r"Qt\.Vertical\b",     "Qt.Vertical → Qt.Orientation.Vertical"),
        (r"QSlider\.TicksBelow\b", "QSlider.TicksBelow → QSlider.TickPosition.TicksBelow"),
    ]
    import os
    widget_src = open(
        os.path.join(os.path.dirname(__file__), "simpli_neighbors_widget.py")
    ).read()
    hits = []
    for pattern, msg in pyqt5_patterns:
        if re.search(pattern, widget_src):
            hits.append(msg)
    assert not hits, "PyQt5 enum(s) found:\n  " + "\n  ".join(hits)

test("no PyQt5-style flat enum references in widget file", test_no_pyqt5_enums)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed:
    sys.exit(1)
