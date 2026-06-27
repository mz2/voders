"""Pure-CPU score-domain augmentation pre-processor (feature 003).

This package fans out *score variants* from each base score along three axes — semitone/octave
transposition, time humanisation (seeded onset/duration jitter), and per-note volume variation —
*before* anything renders. Each variant is just another input score: it flows through the entire
existing 001 pipeline unchanged (parse → render → validate → write), so its labels are correct by
construction (a variant's note rows *are* its labels).

The whole feature is deterministic ``numpy``/integer arithmetic: importing ``voders.scoreaug`` (or
any of its submodules) MUST NOT pull in ``torch`` or any GPU module at module-load time. There is no
model, no out-of-process backend, and no GPU anywhere in this feature.
"""

from __future__ import annotations
