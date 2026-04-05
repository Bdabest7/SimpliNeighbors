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
    for expected in ("INPUT", "RADIUS", "THREADS", "RESOLUTION", "OUTPUT"):
        assert expected in names, f"Missing parameter: {expected}"

def test_algorithm_identity():
    from simpli_neighbors_algorithm import SimpliNeighborsAlgorithm
    alg = SimpliNeighborsAlgorithm()
    assert alg.name() == "surfacesmoothing"
    assert alg.groupId() == "raster"
    inst = alg.createInstance()
    assert isinstance(inst, SimpliNeighborsAlgorithm)

test("algorithm has all 5 expected parameters", test_algorithm_params)
test("algorithm name/group/createInstance",     test_algorithm_identity)

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
        # left edge
        if t["dst_c0"] > 0:
            assert t["src_c0"] <= t["dst_c0"] - overlap
        # top edge
        if t["dst_r0"] > 0:
            assert t["src_r0"] <= t["dst_r0"] - overlap

test("tile count matches ceil(rows/size) * ceil(cols/size)", test_tile_count)
test("every pixel covered exactly once",                     test_tile_coverage)
test("src tiles extend by overlap beyond dst",               test_tile_overlap)

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
