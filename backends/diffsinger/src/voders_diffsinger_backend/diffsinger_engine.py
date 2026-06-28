"""Real neural singing via DiffSinger (out-of-process, ONNX acoustic + NSF-HiFiGAN vocoder).

The acoustic ONNX takes an explicit tensor contract:

    tokens     int64 [1, P]        phoneme ids (TIGER 116-phone inventory)
    durations  int64 [1, P]        per-phoneme length IN MEL FRAMES (fps = sr/hop = 44100/512)
    f0         float [1, T]        per-frame fundamental in Hz, T = sum(durations)
    gender     float [1, T]        key-shift embed (0 = neutral)
    velocity   float [1, T]        speed embed (1 = neutral)
    spk_embed  float [1, T, 256]   per-frame speaker (voice-colour) embedding
    depth      float scalar        diffusion depth (<= max_depth)
    steps      int64 scalar        diffusion sampling steps (speed/quality)
  -> mel       float [1, T, 128]

then NSF-HiFiGAN vocoder: (mel, f0) -> waveform.

Because f0 and per-phoneme durations are INPUTS, score pitch and timing are exact by construction —
no WORLD pitch-relock like the nnsvs lane needs. We place each note's vowel on the beat (the vowel
fills the note, leading/trailing consonants take a tiny frame budget) so the validator's energy
onset stays on the score grid. Multilingual lyrics are handled upstream: the core's espeak G2P emits
per-note IPA phonemes per ``language``; here we map IPA -> the TIGER inventory (English-centric, so
non-English phones approximate to the nearest phone, same spirit as the nnsvs kana mapping).
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

# Repo layout: backends/diffsinger/src/voders_diffsinger_backend/diffsinger_engine.py
# -> repo root is five parents up; bundled models live under models/diffsinger/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODELS = _REPO_ROOT / "models" / "diffsinger"
DEFAULT_MODEL_DIR = _MODELS / "tiger"
DEFAULT_SPEAKER = "tiger_fresh"
_VOCODER_DIR = _MODELS / "nsf_hifigan"

# Diffusion controls. steps trades speed for quality; depth must stay <= the model's max_depth.
DEFAULT_STEPS = 50
DEFAULT_DEPTH = 0.6
# Per-consonant frame budget (at fps≈86, 2 frames ≈ 23 ms). Leading consonants delay the vowel
# onset, so the total lead budget is capped well under the validator's 50 ms onset tolerance.
_CONS_FRAMES = 2
_MAX_LEAD_FRAMES = 3
# Gaps up to this are treated as legato (sustain the previous note) rather than an SP rest, so the
# model holds the note to the score offset instead of releasing early.
_LEGATO_GAP_S = 0.12

# --- IPA (espeak, from the core's phoneme payload) -> TIGER ARPABET-plus inventory --------------
_IPA_VOWEL = {
    "i": "iy", "iː": "iy", "ɪ": "ih", "e": "ey", "ɛ": "eh", "æ": "ae",
    "a": "aa", "ɑ": "aa", "ɑː": "aa", "ʌ": "ah", "ɔ": "ao", "ɒ": "ao",
    "o": "ow", "oʊ": "ow", "ʊ": "uh", "u": "uw", "uː": "uw", "ə": "ax",
    "ɚ": "er", "ɝ": "er", "ɜ": "er", "y": "iy", "ø": "eh", "œ": "eh",
}
_IPA_CONS = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "ɡ": "g", "g": "g",
    "m": "m", "n": "n", "ŋ": "ng", "f": "f", "v": "v", "θ": "th", "ð": "dh",
    "s": "s", "z": "z", "ʃ": "sh", "ʒ": "zh", "h": "hh", "l": "l", "ɫ": "l",
    "r": "r", "ɹ": "r", "w": "w", "j": "y", "tʃ": "ch", "dʒ": "jh",
}
# Plain romaji-ish vowel letters -> ARPABET, for the lyrics-only path (no IPA payload).
_LETTER_VOWEL = {"a": "aa", "e": "eh", "i": "iy", "o": "ow", "u": "uw"}
_REST = "SP"
_DEFAULT_VOWEL = "aa"


@functools.cache
def _inventory(model_dir: str) -> dict[str, int]:
    """phonemes.txt (one phone per line) -> {phone: token_id}."""
    path = Path(model_dir) / "dsacoustic" / "phonemes.txt"
    return {ln.strip(): i for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines())}


def _mel_base(value: object) -> float:
    """dsconfig mel_base is 'e' (natural log) or a number like '10' (log base 10)."""
    import math

    s = str(value).strip().lower()
    return math.e if s == "e" else float(s)


@functools.cache
def _acoustic_config(model_dir: str) -> dict:
    import yaml

    cfg = yaml.safe_load((Path(model_dir) / "dsacoustic" / "dsconfig.yaml").read_text())
    return {
        "sample_rate": int(cfg.get("sample_rate", 44100)),
        "hop_size": int(cfg.get("hop_size", 512)),
        "num_mel_bins": int(cfg.get("num_mel_bins", 128)),
        "max_depth": float(cfg.get("max_depth", 0.6)),
        "mel_base": _mel_base(cfg.get("mel_base", "e")),
    }


@functools.cache
def _vocoder_mel_base() -> float:
    """The vocoder's expected log-mel base (from its vocoder.yaml; default natural log)."""
    import yaml

    cfgs = sorted(_VOCODER_DIR.glob("vocoder.yaml")) + sorted(_VOCODER_DIR.glob("*.yaml"))
    for c in cfgs:
        data = yaml.safe_load(c.read_text()) or {}
        if "mel_base" in data:
            return _mel_base(data["mel_base"])
    return _mel_base("e")


def list_speakers(model_dir: str | Path = DEFAULT_MODEL_DIR) -> list[str]:
    """Voice-colour embeddings bundled with the voicebank (each a 256-float ``*.emb``)."""
    return sorted(p.stem for p in (Path(model_dir) / "dsacoustic").glob("*.emb"))


@functools.cache
def _speaker_embed(model_dir: str, speaker: str) -> np.ndarray:
    """Load a 256-d voice-colour embedding; fall back to the default speaker, then zeros."""
    acou = Path(model_dir) / "dsacoustic"
    for name in (speaker, DEFAULT_SPEAKER):
        emb = acou / f"{name}.emb"
        if emb.exists():
            return np.fromfile(emb, dtype=np.float32)
    any_emb = sorted(acou.glob("*.emb"))
    if any_emb:
        return np.fromfile(any_emb[0], dtype=np.float32)
    return np.zeros(256, dtype=np.float32)


def _map_vowel(ph: str) -> str:
    if ph in _IPA_VOWEL:
        return _IPA_VOWEL[ph]
    base = ph[0] if ph else ""
    return _IPA_VOWEL.get(base, _LETTER_VOWEL.get(base, _DEFAULT_VOWEL))


def _map_cons(ph: str) -> str | None:
    return _IPA_CONS.get(ph) or _IPA_CONS.get(ph[:1])


def _syllable_vowel(syllable: str | None) -> str:
    if not syllable:
        return _DEFAULT_VOWEL
    for c in syllable.lower():
        if c in _LETTER_VOWEL:
            return _LETTER_VOWEL[c]
    return _DEFAULT_VOWEL


def _note_phones(entry: dict | None, syllable: str | None) -> tuple[list[str], str, list[str]]:
    """Resolve a note to (leading consonants, vowel, trailing consonants) in the TIGER inventory.

    Prefers the core's language-aware IPA payload (``entry``); falls back to the lyric syllable's
    vowel; finally an open vowel. This is where multilingual lyrics enter the model.
    """
    if entry is not None:
        nucleus = next((p for p in entry.get("phonemes", []) if _map_vowel(p) and p not in _IPA_CONS), None)
        vowel = _map_vowel(nucleus) if nucleus else _syllable_vowel(syllable)
        lead = [m for m in (_map_cons(p) for p in entry.get("lead", [])) if m]
        tail = [m for m in (_map_cons(p) for p in entry.get("tail", [])) if m]
        return lead, vowel, tail
    return [], _syllable_vowel(syllable), []


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _build_sequence(
    notes: list[list[float]],
    phonemes: list[dict] | None,
    lyrics: list[str | None] | None,
    fps: float,
    ph2id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (tokens, durations[frames], f0[per-frame]) with the vowel of each note on the beat.

    Rests (gaps, and the lead-in before the first note) are SP tokens with f0=0; each note's vowel
    starts at round(onset*fps) so the energy onset lands on the score grid (label-aligned).
    """
    by_note = {int(p["note_index"]): p for p in (phonemes or [])}
    syls = list(lyrics or [])
    tokens: list[int] = []
    durs: list[int] = []
    f0_per_ph: list[float] = []

    def emit(phone: str, frames: int, hz: float) -> None:
        if frames <= 0:
            return
        tokens.append(ph2id.get(phone, ph2id.get(_REST, 0)))
        durs.append(int(frames))
        f0_per_ph.append(hz)

    # The lead-in and every inter-note gap are emitted by the gap check below (prev_off starts at 0),
    # so note 0's vowel lands exactly at its score onset. A separate lead-in rest here would double
    # the pre-roll and shift every note late (failing the onset gate).
    prev_off = 0.0
    for i, (onset, offset, pitch) in enumerate(notes):
        gap = onset - prev_off
        if gap > 1e-3:
            gap_frames = max(1, round(gap * fps))
            # A short gap mid-phrase is legato, not a rest: inserting SP makes the model release the
            # previous note early (its offset decays ~100 ms before the score grid). Instead sustain
            # the previous note's last phoneme through the gap so its offset lands on the grid. Only a
            # real (long) gap, or the lead-in before note 0, becomes an SP rest.
            if i > 0 and gap <= _LEGATO_GAP_S and durs:
                durs[-1] += gap_frames
            else:
                emit(_REST, gap_frames, 0.0)
        n_frames = max(1, round((offset - onset) * fps))
        entry = by_note.get(i)
        syl = syls[i] if i < len(syls) else None
        lead, vowel, tail = _note_phones(entry, syl)
        hz = _midi_to_hz(pitch)

        # Frame budget: tiny lead/tail consonants, vowel fills the rest so its onset stays on-beat.
        lead_n = min(_MAX_LEAD_FRAMES, _CONS_FRAMES * len(lead))
        tail_n = min(_CONS_FRAMES * len(tail), max(0, n_frames - lead_n - 1))
        vowel_n = max(1, n_frames - lead_n - tail_n)
        if lead:
            per = max(1, lead_n // len(lead))
            for c in lead[:-1]:
                emit(c, per, hz)
            emit(lead[-1], max(1, lead_n - per * (len(lead) - 1)), hz)
        emit(vowel, vowel_n, hz)
        if tail:
            per = max(1, tail_n // len(tail))
            for c in tail[:-1]:
                emit(c, per, hz)
            emit(tail[-1], max(1, tail_n - per * (len(tail) - 1)), hz)
        prev_off = offset
    emit(_REST, max(1, round(0.2 * fps)), 0.0)  # trailing rest

    tokens_a = np.asarray(tokens, dtype=np.int64)
    durs_a = np.asarray(durs, dtype=np.int64)
    f0 = np.repeat(np.asarray(f0_per_ph, dtype=np.float32), durs_a)
    return tokens_a, durs_a, f0


@functools.cache
def _preload_cuda_libs() -> None:
    """Pull cuDNN 9 from the ``nvidia-cudnn-cu13`` wheel (the only CUDA lib not on the system CUDA 13
    toolkit) via onnxruntime's loader. The CUDA-13.1 forward-compat libcuda (cuda-compat-13-1) is
    selected *before* the process starts, via ``LD_LIBRARY_PATH`` injected by the core's
    backend_bridge — doing it in-process (ctypes) races onnxruntime's own libcuda load and hangs."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:  # noqa: BLE001
            pass


def _providers() -> list[str]:
    """CUDA when the GPU build is present (the GB10 target), else CPU for a plain onnxruntime build.

    No runtime CUDA->CPU self-heal: a CUDA execution failure is raised so it's visible rather than
    silently producing CPU output (the corpus must know which device rendered)."""
    import onnxruntime as ort

    _preload_cuda_libs()
    avail = set(ort.get_available_providers())
    return (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else []) + [
        "CPUExecutionProvider"
    ]


@functools.cache
def _session(path: str):
    _preload_cuda_libs()  # compat libcuda + cuDNN, before onnxruntime first touches CUDA
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # quiet
    return ort.InferenceSession(path, sess_options=opts, providers=_providers())


@functools.cache
def _vocoder_path() -> str:
    onnx = sorted(_VOCODER_DIR.glob("*.onnx"))
    if not onnx:
        raise RuntimeError(f"no NSF-HiFiGAN .onnx under {_VOCODER_DIR}")
    return str(onnx[0])


def _infer(acoustic_path: str, feeds: dict, f0: np.ndarray, mel_base_factor: float) -> np.ndarray:
    """Run acoustic -> vocoder (same EP), converting the mel to the vocoder's log base in between.

    ``mel_base_factor = ln(acoustic_base)/ln(vocoder_base)`` rescales log_b1(S) -> log_b2(S). The
    TIGER acoustic emits log10-mel but the 2025.02 NSF-HiFiGAN expects ln-mel; skipping this makes
    the vocoder breathy/aperiodic (clear spectral f0 but pyin sees it as unvoiced -> validator 0%)."""
    acoustic = _session(acoustic_path)
    fed = {k: v for k, v in feeds.items() if k in {i.name for i in acoustic.get_inputs()}}
    mel = acoustic.run(["mel"], fed)[0]
    if abs(mel_base_factor - 1.0) > 1e-6:
        mel = (mel * mel_base_factor).astype(np.float32)
    vocoder = _session(_vocoder_path())
    out_name = vocoder.get_outputs()[0].name
    return vocoder.run([out_name], {"mel": mel, "f0": f0})[0].reshape(-1)


def render_diffsinger(
    notes: list[list[float]],
    sr: int,
    *,
    phonemes: list[dict] | None = None,
    lyrics: list[str | None] | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    speaker: str = DEFAULT_SPEAKER,
    steps: int = DEFAULT_STEPS,
    depth: float = DEFAULT_DEPTH,
) -> np.ndarray:
    """Render score-exact DiffSinger singing for ``notes`` (+ phonemes/lyrics), resampled to ``sr``."""
    from scipy.signal import resample_poly

    model_dir = str(model_dir)
    cfg = _acoustic_config(model_dir)
    fps = cfg["sample_rate"] / cfg["hop_size"]
    ph2id = _inventory(model_dir)

    tokens, durs, f0 = _build_sequence(notes, phonemes, lyrics, fps, ph2id)
    n_frames = int(durs.sum())
    spk = np.tile(_speaker_embed(model_dir, speaker).reshape(1, 1, -1), (1, n_frames, 1)).astype(
        np.float32
    )
    feeds = {
        "tokens": tokens[None, :],
        "durations": durs[None, :],
        "f0": f0[None, :],
        "gender": np.zeros((1, n_frames), dtype=np.float32),
        "velocity": np.ones((1, n_frames), dtype=np.float32),
        "spk_embed": spk,
        "depth": np.asarray(min(depth, cfg["max_depth"]), dtype=np.float32),
        "steps": np.asarray(int(steps), dtype=np.int64),
    }
    import math

    mel_base_factor = math.log(cfg["mel_base"]) / math.log(_vocoder_mel_base())
    acoustic_path = str(Path(model_dir) / "dsacoustic" / "acoustic.onnx")
    wav = _infer(acoustic_path, feeds, f0[None, :], mel_base_factor)

    wav = np.asarray(wav, dtype=np.float64)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav / peak * 0.9
    model_sr = cfg["sample_rate"]
    if model_sr != sr:
        wav = resample_poly(wav, sr, model_sr)
    return wav.astype(np.float32)
