"""Project-wide constants fixed by the consumer contract (FR-002, Assumptions).

The downstream consumer is a Basic Pitch-class transcription model that ingests
22,050 Hz mono float32 audio, so every rendered sample uses this format.
"""

from __future__ import annotations

SAMPLE_RATE: int = 22_050
"""Audio sample rate in Hz (Basic Pitch's required input)."""

ONSET_TOLERANCE_MS: float = 50.0
"""Onset tolerance window (SC-001); also the hard floor for the learned min_note_ms."""

OFFSET_MIN_MS: float = 50.0
"""Lower bound of the offset tolerance; the rule is max(50 ms, 20% of note length) (SC-002)."""

OFFSET_FRACTION: float = 0.20
"""Fraction-of-note-length component of the offset tolerance (SC-002)."""

F0_CENTS_TOLERANCE: float = 25.0
"""Measured f0 must stay within ±25 cents of the score pitch (US1 scenario 2)."""

F0_COVERAGE: float = 0.80
"""Fraction of a note's sustained interval that must satisfy the f0 tolerance."""

EXPRESSIVE_ONSET_DEV_MS: float = 50.0
"""Re-derived onset deviation bound for the expressive lane (SC-010)."""
