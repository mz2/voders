"""RendererLane interface contract (FR-015; contract: contracts/lane-interface.md).

The renderer boundary that makes the four lanes independently enable/disable/replaceable. The
orchestrator depends only on this Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from voders.scores.models import Score
from voders.voices.models import Voice


@dataclass
class RenderRequest:
    """Input to a lane's ``render`` (contract: lane-interface.md)."""

    score: Score
    voice: Voice
    seed: int
    options: dict[str, object] = field(default_factory=dict)
    # Optional per-note syllables resolved by the lyric layer (FR-006). ``None`` (or a ``None``
    # entry) means open-vowel for that note. Only the SVS lane articulates; other lanes ignore them.
    lyrics: list[str | None] | None = None
    # G2P backend the SVS lane uses to phonemize syllables: "espeak" (CPU rules) | "neural" (GPU).
    g2p_backend: str = "espeak"
    # espeak language code for G2P/articulation (multilingual phonetic coverage).
    language: str = "en-us"


@dataclass
class RenderResult:
    """Output of a lane's ``render``.

    ``audio`` is 22,050 Hz mono float32 (FR-002). ``label_score`` is the score the audio is
    labelled by — the input score, or a re-derived score for the expressive ``rederive_labels``
    mode (FR-007). ``notes`` carries lane diagnostics merged into the verdict (e.g.
    ``max_onset_dev_ms``).
    """

    audio: np.ndarray
    label_score: Score
    notes: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class RendererLane(Protocol):
    """The contract every lane implements (FR-015)."""

    name: str

    def requires_gpu(self) -> bool:
        """Deterministic lane and the validator MUST return False (FR-009)."""
        ...

    def render(self, req: RenderRequest) -> RenderResult: ...
