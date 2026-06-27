"""CPU lyric layer: per-note syllable assignment and articulation (feature 002).

This package is import-safe on a CPU-only baseline: importing ``voders.lyrics`` (or its
``models``/``sources`` submodules) MUST NOT pull in ``torch`` or ``phonemizer`` at module load
(FR-005). Heavy/optional dependencies are imported lazily inside the functions that need them
(e.g. the G2P backend and the out-of-process ``generated`` source).
"""

from __future__ import annotations
