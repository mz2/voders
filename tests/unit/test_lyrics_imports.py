"""Import-guard tests for the CPU lyric layer (T002, FR-005).

Mirrors ``tests/unit/test_device.py``: the lyric package and its data-carrier/source submodules
must be import-safe on a CPU-only baseline and must never load ``torch`` or ``phonemizer`` at
module-load time. Heavy backends are imported lazily inside the functions that use them.
"""

from __future__ import annotations

import importlib
import sys


def _reimport(module_name: str) -> None:
    """Re-import ``module_name`` fresh and assert no heavy deps appear in ``sys.modules``."""
    for mod in (module_name, "torch", "phonemizer"):
        sys.modules.pop(mod, None)
    importlib.import_module(module_name)
    assert "torch" not in sys.modules, f"{module_name} must not import torch (FR-005)"
    assert "phonemizer" not in sys.modules, f"{module_name} must not import phonemizer (FR-005)"


def test_import_lyrics_package_is_cpu_safe():
    _reimport("voders.lyrics")


def test_import_lyrics_models_is_cpu_safe():
    _reimport("voders.lyrics.models")


def test_import_lyrics_sources_is_cpu_safe():
    _reimport("voders.lyrics.sources")
