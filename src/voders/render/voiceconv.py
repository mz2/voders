"""Voice-conversion (timbre) lane (FR-004, US2; research Decision 4).

Voice conversion ("VC") changes a singer's *timbre* while keeping the score-derived pitch contour
(fundamental frequency, "f0") unchanged, so onset/offset/pitch alignment is preserved and
``label_score == req.score``.

Backends (``options["backend"]``):

* ``"world"`` (default, CPU): delegates to the deterministic WORLD lane, which builds f0 directly
  from the score and uses the voice's donor recording as the timbre source. No GPU, no ``torch``.
* ``"rvc"``: the real RVC toolkit, run **out of process** in its own uv project (``backends/rvc``,
  Python 3.10) via :mod:`voders.render.backend_bridge`. The core renders score-aligned audio with
  WORLD, then RVC converts only the timbre using a *consented* target voice model
  (``voice.model_ref``, an RVC ``.pth``); RVC keeps the input audio's f0, so the labels are
  preserved (``auto_predict_f0=False`` equivalent).
* ``"sovits"``: so-vits-svc — a GitHub repo (not a PyPI package); wire it as another backend
  project when its weights are available.
"""

from __future__ import annotations

import numpy as np

from voders.constants import SAMPLE_RATE
from voders.render.base import RenderRequest, RenderResult
from voders.render.deterministic import midi_to_hz
from voders.scores.models import Score
from voders.voices.models import Voice, VoiceKind

_MAX_LATENCY_S = 0.3  # search window for the converter's processing latency


def _pitch_correct(audio: np.ndarray, score: Score, strength: float) -> np.ndarray:
    """Subtle per-note pitch centring: shift each note's f0 contour so its *median* sits on the
    score pitch, keeping the contour shape (vibrato, scoops) intact.

    ``strength`` dials it: 0 = off, 1 = each note fully centred on its pitch. Because it *shifts*
    the contour rather than flattening it to a constant, it corrects audible drift without the hard
    autotune ("Cher") sound. Uses WORLD analysis/resynthesis but keeps the converted timbre (its
    spectral envelope and aperiodicity are untouched — only f0 moves).
    """
    if strength <= 0:
        return audio
    import pyworld

    x = np.ascontiguousarray(audio, dtype=np.float64)
    if x.size < SAMPLE_RATE // 50:
        return audio
    f0, t = pyworld.harvest(x, SAMPLE_RATE)
    sp = pyworld.cheaptrick(x, f0, t, SAMPLE_RATE)
    ap = pyworld.d4c(x, f0, t, SAMPLE_RATE)
    f0c = f0.copy()
    for note in score.notes:
        m = (t >= note.onset_s) & (t < note.offset_s) & (f0 > 0)
        if int(m.sum()) < 3:
            continue
        med = float(np.median(f0[m]))
        if med <= 0:
            continue
        f0c[m] = f0[m] * (midi_to_hz(note.pitch_midi) / med) ** float(strength)
    y = pyworld.synthesize(f0c, sp, ap, SAMPLE_RATE).astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = (y / peak * 0.98).astype(np.float32)
    return y


def _align_to_source(converted: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift ``converted`` to compensate the converter's latency, aligning it to ``source``.

    ``source`` is the exact-grid WORLD render (its onsets ARE the labels). A generative converter
    adds a near-constant latency, so we estimate the global lag by cross-correlating the two energy
    envelopes and shift ``converted`` back onto the grid, then match length and de-clip. Returns the
    re-aligned audio and the compensated latency in milliseconds.
    """
    c = np.asarray(converted, dtype=np.float32)
    s = np.asarray(source, dtype=np.float32)
    n = min(c.size, s.size)
    if n < SAMPLE_RATE // 10:
        return c, 0.0
    ds = max(1, SAMPLE_RATE // 2000)  # downsample envelopes to ~2 kHz for a cheap correlation
    win = max(1, SAMPLE_RATE // 200)  # ~5 ms smoothing
    kernel = np.ones(win, dtype=np.float64) / win

    def _env(x: np.ndarray) -> np.ndarray:
        e = np.convolve(np.abs(x[:n]).astype(np.float64), kernel, mode="same")[::ds]
        return e - e.mean()

    ec, es = _env(c), _env(s)
    corr = np.correlate(ec, es, mode="full")
    mid = es.size - 1
    span = int(_MAX_LATENCY_S * SAMPLE_RATE) // ds
    lo, hi = max(0, mid - span), min(corr.size - 1, mid + span)
    lag = (int(np.argmax(corr[lo : hi + 1])) + lo - mid) * ds  # >0: converted lags the source

    if lag > 0:
        c = c[lag:]
    elif lag < 0:
        c = np.concatenate([np.zeros(-lag, dtype=np.float32), c])
    if c.size < s.size:
        c = np.concatenate([c, np.zeros(s.size - c.size, dtype=np.float32)])
    else:
        c = c[: s.size]
    peak = float(np.max(np.abs(c))) if c.size else 0.0
    if peak > 1.0:
        c = (c / peak * 0.98).astype(np.float32)
    return c, round(lag * 1000.0 / SAMPLE_RATE, 1)


_CPU_BACKENDS = frozenset({"world"})
# Out-of-process backends → (backend project dir under backends/, worker module). Each runs the
# real toolkit in its own uv project. "rvc" uses a trained .pth; "seedvc" is zero-shot (model_ref
# is a reference audio clip — e.g. a consented VocalSet/VCTK donor).
_SUBPROCESS_BACKENDS = {
    "rvc": ("rvc", "voders_rvc_backend.worker"),
    "seedvc": ("seedvc", "voders_seedvc_backend.worker"),
}
_REPO_BACKENDS = frozenset({"sovits"})  # GitHub repo + weights, not yet wired


class VoiceConversionLane:
    """Timbre fan-out lane keeping the score f0 (FR-004)."""

    name = "voice_conversion"

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}
        self.backend = str(self.options.get("backend", "world"))

    def requires_gpu(self) -> bool:
        """False: ``world`` is CPU and ``rvc`` runs out-of-process, so the core needs no GPU."""
        return False

    def render(self, req: RenderRequest) -> RenderResult:
        backend = str(req.options.get("backend", self.backend))
        from voders.render.deterministic import DeterministicLane

        if backend in _CPU_BACKENDS:
            # WORLD constructs f0 from the score (never predicted/altered) using the donor timbre.
            result = DeterministicLane().render(req)
            return RenderResult(
                audio=result.audio, label_score=req.score, notes={"backend": backend}
            )

        if backend in _SUBPROCESS_BACKENDS:
            if not req.voice.model_ref:
                raise RuntimeError(
                    f"voice-conversion backend {backend!r} requires a consented target voice "
                    "(voice.model_ref: an RVC .pth, or a reference clip for zero-shot seedvc) "
                    "(FR-011)"
                )
            base_donor = req.options.get("base_donor") or self.options.get("base_donor")
            if not base_donor:
                raise RuntimeError(
                    f"voice-conversion backend {backend!r} needs a 'base_donor' (a donor vowel "
                    "wav) for the WORLD source render that the converter then re-timbres"
                )
            from voders.render.backend_bridge import convert_via_backend

            # 1) WORLD renders score-aligned source audio from a donor vowel (timbre irrelevant —
            #    the converter replaces it); 2) the toolkit converts to the target voice, keeping
            #    its f0 so the labels are preserved.
            project, module = _SUBPROCESS_BACKENDS[backend]
            donor = Voice(
                voice_id=f"{req.voice.voice_id}__base",
                kind=VoiceKind.DETERMINISTIC_DONOR,
                consent_verified=True,
                model_ref=str(base_donor),
            )
            base = (
                DeterministicLane()
                .render(
                    RenderRequest(score=req.score, voice=donor, seed=req.seed, options=req.options)
                )
                .audio
            )
            params: dict[str, object] = {}
            if backend == "seedvc" and "diffusion_steps" in req.options:
                params["diffusion_steps"] = req.options["diffusion_steps"]
            converted = convert_via_backend(
                project,
                module,
                base,
                model_ref=req.voice.model_ref,
                device=str(req.options.get("device", "auto")),
                params=params,
            )
            # Generative converters (seedvc) introduce a near-constant processing latency, so onsets
            # drift off the score grid even though f0 is preserved. Re-align the converted audio to
            # the exact-grid WORLD source by global cross-correlation — keeps the human timbre while
            # snapping onsets back onto the labels (FR-004).
            converted, lag_ms = _align_to_source(converted, base)
            # Optional subtle pitch correction: pull each note's median f0 onto the score pitch
            # while keeping the contour, so generative drift is tuned out without flattening.
            strength = float(
                req.options.get("pitch_correct", self.options.get("pitch_correct", 0.0))
            )
            corrected = _pitch_correct(converted, req.score, strength)
            if corrected.size:
                converted = corrected
            return RenderResult(
                audio=converted,
                label_score=req.score,
                notes={"backend": backend, "vc_latency_ms": lag_ms, "pitch_correct": strength},
            )

        if backend in _REPO_BACKENDS:
            raise RuntimeError(
                f"voice-conversion backend {backend!r} is a GitHub repo + weights; "
                "wire it as a backend project under backends/ before use"
            )

        raise ValueError(
            f"unknown voice-conversion backend {backend!r}; expected one of "
            f"{sorted(_CPU_BACKENDS | set(_SUBPROCESS_BACKENDS) | _REPO_BACKENDS)}"
        )
