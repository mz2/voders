"""Expressive neural singing-voice-synthesis ("SVS") lane with a label safety net (FR-007, US4).

The expressive lane adds naturalistic timing (vibrato, slight onset humanization) that a fixed,
score-derived render does not. That expressiveness is exactly what can push a note's onset outside
the challenge's 50 ms tolerance and silently corrupt the label. The lane therefore exposes two
safety modes (FR-007, research Decision 3):

* ``force_score_f0`` (default): use the score's f0 (fundamental frequency, the sung pitch) directly,
  so onsets/offsets/pitch are correct by construction and ``label_score == req.score`` (exact).
* ``rederive_labels``: render *expressively* by humanizing the note timing with a seeded jitter,
  then RE-DERIVE the note onsets/offsets from the actually-rendered audio (a forced-alignment /
  onset-detection analogue) and carry the re-derived score; if any onset drifts past 50 ms the
  sample is forced REJECTED so the orchestrator drops it (SC-002).

Backends (``options["backend"]``):

* ``"cpu"`` (default): an expressive WORLD-vocoder stand-in, in-process, no GPU.
* ``"nnsvs"``: the NNSVS toolkit, run **out of process** in its own uv project
  (``backends/svs``, Python 3.11) via :mod:`voders.render.backend_bridge`, because its dependency
  chain conflicts with the 3.14 core.
* ``"diffsinger"``: a production in-process GPU backend; ``torch`` is imported lazily inside
  ``render`` (FR-009) and, since the toolkit/weights are not wired here, ``render`` raises.
"""

from __future__ import annotations

import numpy as np

from voders.render.base import RenderRequest, RenderResult
from voders.scores.models import Note, Score
from voders.seeds import rng

_CPU_BACKENDS = frozenset({"cpu"})
_SUBPROCESS_BACKENDS = frozenset({"nnsvs"})  # out-of-process, own uv project
_GPU_BACKENDS = frozenset({"diffsinger"})  # in-process GPU
# Backends that can sing real phonemes from per-note syllables (FR-006). The ``cpu`` stand-in sings
# an open vowel and does NOT articulate, so it never sets ``lyric_articulated``.
_ARTICULATING_BACKENDS = _SUBPROCESS_BACKENDS | _GPU_BACKENDS
_DEFAULT_HUMANIZE_MS = 20.0
_ONSET_TOLERANCE_MS = 50.0  # SC-002


class SvsLane:
    """Expressive SVS lane with ``force_score_f0`` / ``rederive_labels`` safety modes (FR-007)."""

    name = "svs"

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}

    def requires_gpu(self) -> bool:
        """True only for in-process GPU backends; ``cpu`` and out-of-process ``nnsvs`` are False."""
        backend = str(self.options.get("backend", "cpu"))
        return backend in _GPU_BACKENDS

    def _opt(self, req: RenderRequest, key: str, default: object) -> object:
        if key in req.options:
            return req.options[key]
        return self.options.get(key, default)

    def render(self, req: RenderRequest) -> RenderResult:
        backend = str(self._opt(req, "backend", "cpu"))
        if backend in _GPU_BACKENDS:
            return self._render_gpu(backend)
        if backend not in _CPU_BACKENDS and backend not in _SUBPROCESS_BACKENDS:
            raise ValueError(
                f"unknown SVS backend {backend!r}; expected one of "
                f"{sorted(_CPU_BACKENDS | _SUBPROCESS_BACKENDS | _GPU_BACKENDS)}"
            )

        mode = str(self._opt(req, "mode", "force_score_f0"))
        if mode == "force_score_f0":
            audio = self._render_audio(req, req.score, backend)
            notes: dict[str, object] = {"mode": mode, "backend": backend}
            if self._articulated(req, backend):
                notes["lyric_articulated"] = True
            return RenderResult(audio=audio, label_score=req.score, notes=notes)
        if mode == "rederive_labels":
            raw = self._opt(req, "humanize_ms", _DEFAULT_HUMANIZE_MS)
            humanize_ms = float(raw) if isinstance(raw, int | float | str) else _DEFAULT_HUMANIZE_MS
            return self._render_rederive(req, backend, humanize_ms)
        raise ValueError(
            f"unknown SVS mode {mode!r}; expected 'force_score_f0' or 'rederive_labels'"
        )

    @staticmethod
    def _articulated(req: RenderRequest, backend: str) -> bool:
        """True only when real phonemes are sung: lyrics present + articulating backend (FR-006).

        The ``cpu`` stand-in sings an open vowel regardless of lyrics, so it never articulates.
        """
        has_lyrics = req.lyrics is not None and any(s for s in req.lyrics)
        return has_lyrics and backend in _ARTICULATING_BACKENDS

    @staticmethod
    def _phoneme_payload(req: RenderRequest, score: Score) -> list[dict] | None:
        """Core-side G2P (espeak) -> per-note phoneme runs for the backend to articulate (L3).

        Returns ``None`` when there are no lyrics (open-vowel render). Lazily imports the G2P module
        so the CPU baseline never loads phonemizer/espeak (FR-005).
        """
        if not (req.lyrics is not None and any(s for s in req.lyrics)):
            return None
        from voders.lyrics.g2p import phoneme_runs

        runs = phoneme_runs(req.lyrics, score, backend=req.g2p_backend)
        return [
            {
                "note_index": r.note_index,
                "phonemes": r.phonemes,
                "lead": r.lead_consonants,
                "tail": r.tail_consonants,
                "nucleus_onset_s": r.nucleus_onset_s,
            }
            for r in runs
        ]

    def _render_audio(self, req: RenderRequest, score: Score, backend: str) -> np.ndarray:
        """Render audio for ``score`` with the selected backend (in-process or out-of-process)."""
        if backend in _SUBPROCESS_BACKENDS:
            from voders.render.backend_bridge import render_via_backend

            return render_via_backend(
                "svs",
                "voders_svs_backend.worker",
                score,
                req.seed,
                model_ref=req.voice.model_ref,
                lyrics=req.lyrics,
                phonemes=self._phoneme_payload(req, score),
            )
        from voders.render.deterministic import DeterministicLane

        return (
            DeterministicLane()
            .render(RenderRequest(score=score, voice=req.voice, seed=req.seed, options=req.options))
            .audio
        )

    def _render_rederive(
        self, req: RenderRequest, backend: str, humanize_ms: float
    ) -> RenderResult:
        from voders.validate.rederive import derive_labels

        jittered = _humanize_timing(req.score, req.seed, humanize_ms)
        audio = self._render_audio(req, jittered, backend)
        rederived, max_dev_ms = derive_labels(audio, req.score)

        notes: dict[str, object] = {
            "mode": "rederive_labels",
            "backend": backend,
            "humanize_ms": humanize_ms,
            "max_onset_dev_ms": max_dev_ms,
        }
        if self._articulated(req, backend):
            notes["lyric_articulated"] = True
        if max_dev_ms > _ONSET_TOLERANCE_MS:
            notes["force_status"] = "rejected"
            notes["reject_reason"] = (
                f"re-derived onset deviation {max_dev_ms:.1f} ms exceeds "
                f"{_ONSET_TOLERANCE_MS:.0f} ms tolerance (SC-002)"
            )
        return RenderResult(audio=audio, label_score=rederived, notes=notes)

    @staticmethod
    def _render_gpu(backend: str) -> RenderResult:
        import importlib

        try:
            importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                f"SVS backend {backend!r} requires the 'gpu' extra (torch is not installed)"
            ) from exc
        raise RuntimeError(
            f"SVS backend {backend!r} requires the GPU toolkit (install the 'gpu' extra and the "
            f"{backend} model weights); not available here"
        )


def _humanize_timing(score: Score, seed: int, humanize_ms: float) -> Score:
    """Shift each note by a small seeded jitter, keeping its duration (FR-013 reproducible)."""
    gen = rng(seed)
    jittered: list[Note] = []
    for note in score.notes:
        frac = float(gen.uniform(0.5, 1.0))
        sign = 1.0 if gen.random() < 0.5 else -1.0
        delta_s = sign * frac * humanize_ms / 1000.0
        onset = max(0.0, note.onset_s + delta_s)
        offset = onset + note.duration_s
        jittered.append(Note(onset_s=onset, offset_s=offset, pitch_midi=note.pitch_midi))
    return Score(score_id=score.score_id, source=score.source, notes=jittered)
