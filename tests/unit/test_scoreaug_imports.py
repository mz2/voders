"""Import-guard tests for the pure-CPU score-augmentation package (T001).

Mirrors ``tests/unit/test_lyrics_imports.py``: the score-augmentation package and its submodules
must be import-safe on a CPU-only baseline and must never load ``torch`` (or any GPU module) at
module-load time. The whole feature is deterministic ``numpy``/integer arithmetic.
"""

from __future__ import annotations

import importlib
import sys


def _reimport(module_name: str) -> None:
    """Re-import ``module_name`` fresh and assert no heavy GPU deps appear in ``sys.modules``."""
    for mod in (module_name, "torch"):
        sys.modules.pop(mod, None)
    importlib.import_module(module_name)
    assert "torch" not in sys.modules, (
        f"{module_name} must not import torch (feature is numpy-only)"
    )


def test_import_scoreaug_package_is_cpu_safe():
    _reimport("voders.scoreaug")


def test_import_scoreaug_models_is_cpu_safe():
    _reimport("voders.scoreaug.models")


def test_import_scoreaug_expand_is_cpu_safe():
    _reimport("voders.scoreaug.expand")
