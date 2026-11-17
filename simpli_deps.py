# SimpliNeighbors — dependency check (finding 14)
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Pinned versions are in requirements.txt (the ones tested on the VM). Install
# them ahead of time, then let the automation's preflight run this file with
# QGIS's Python:
#
#     python-qgis.bat <plugin folder>\simpli_deps.py [--require numba,scipy]
#
# One line per dependency ("dep <name> <version> ok" / "differs from tested <pin>"
# / "MISSING <why>"); exit code 1 when a required dependency is missing.
# Always required: numpy, scipy, osgeo (GDAL). --require adds numba/llvmlite for
# a run that pins the Numba CPU or CUDA backend.

from __future__ import annotations

import importlib
import os
import sys

# import name -> (pip name, needed by default)
DEPENDENCIES = {
    "numpy":    ("numpy", True),
    "scipy":    ("scipy", True),
    "osgeo":    ("GDAL", True),
    "numba":    ("numba", False),
    "llvmlite": ("llvmlite", False),
}


def _pins() -> dict:
    pins = {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if "==" in line:
                    name, ver = line.split("==", 1)
                    pins[name.strip().lower()] = ver.strip()
    except OSError:
        pass
    return pins


def check_dependencies(require=()):
    """Return (lines, missing_required). `require` names extra dependencies
    (e.g. 'numba') that must be importable for this run."""
    pins = _pins()
    need = {n for n, (_, req) in DEPENDENCIES.items() if req} | {r.strip() for r in require if r.strip()}
    lines, missing = [], []
    for mod, (pkg, _) in DEPENDENCIES.items():
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # ImportError or a broken DLL
            lines.append(f"dep {pkg} MISSING {e}")
            if mod in need:
                missing.append(pkg)
            continue
        version = str(getattr(m, "__version__", "?"))
        want = pins.get(pkg.lower())
        if want and version != want:
            lines.append(f"dep {pkg} {version} differs from tested {want}")
        else:
            lines.append(f"dep {pkg} {version} ok")
    return lines, missing


def require_for(backend: str | None, mode_needs_scipy: bool = True):
    """Raise ImportError naming what a run needs and cannot import (used by the algorithm)."""
    req = []
    if backend in ("CUDA", "Numba CPU"):
        req += ["numba", "llvmlite"]
    _, missing = check_dependencies(req)
    if not mode_needs_scipy:
        missing = [m for m in missing if m != "scipy"]
    if missing:
        pins = _pins()
        hint = " ".join(f"{m}=={pins[m.lower()]}" if m.lower() in pins else m for m in missing)
        raise ImportError(
            "SimpliNeighbors needs " + ", ".join(missing) + " in this Python. Install with: "
            "python -m pip install " + hint)


if __name__ == "__main__":
    extra = []
    if "--require" in sys.argv:
        i = sys.argv.index("--require")
        extra = sys.argv[i + 1].split(",") if i + 1 < len(sys.argv) else []
    out, miss = check_dependencies(extra)
    for line in out:
        print(line)
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    sys.exit(1 if miss else 0)
