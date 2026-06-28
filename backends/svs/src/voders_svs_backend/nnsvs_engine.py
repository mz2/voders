"""Real neural singing via NNSVS (out-of-process, GPU).

Renders a score + per-note syllables to natural singing with a trained NNSVS voicebank, replacing
the formant stand-in. Pipeline: build a MusicXML from the notes (pitch + durations, rests for gaps)
with the syllables mapped to Japanese kana mora, run it through Sinsy (pysinsy) to get HTS
full-context labels, then NNSVS ``SPSVS.svs`` synthesises the waveform on the GPU.

Why kana: NNSVS's available pretrained voicebanks are Japanese. Scat / nonsense syllables map
cleanly onto kana mora (ba, do, ki, bo, ...), so the result sounds like natural singing of those
syllables. The model id is the voicebank (``r9y9/yoko_latest`` by default; selectable per voice).

Heavy deps (torch/nnsvs/pysinsy) are imported lazily so the formant path stays usable without them.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import numpy as np

# Romaji-ish syllable -> hiragana mora. Covers the CV space our inventories use; unknown onsets fall
# back to the nearest row, unknown syllables to a bare vowel. Scat is nonsense, so approx is fine.
_VOWELS = "aiueo"
_MORA = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "sa": "さ", "si": "し", "su": "す", "se": "せ", "so": "そ",
    "ta": "た", "ti": "ち", "tu": "つ", "te": "て", "to": "と",
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "hu": "ふ", "he": "へ", "ho": "ほ",
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ",
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "wa": "わ", "wo": "を",
    "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご",
    "za": "ざ", "zi": "じ", "zu": "ず", "ze": "ぜ", "zo": "ぞ",
    "da": "だ", "di": "ぢ", "du": "づ", "de": "で", "do": "ど",
    "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ",
    "pa": "ぱ", "pi": "ぴ", "pu": "ぷ", "pe": "ぺ", "po": "ぽ",
}
# Map English onset letters to a kana row consonant.
_ROW = {
    "k": "k", "c": "k", "q": "k", "g": "g", "s": "s", "z": "z", "j": "z",
    "t": "t", "d": "d", "n": "n", "h": "h", "f": "h", "b": "b", "p": "p",
    "v": "b", "m": "m", "y": "y", "r": "r", "l": "r", "w": "w",
}
DEFAULT_MODEL = "r9y9/yoko_latest"

# Voice "characters" derived from the base voicebank by WORLD formant (vocal-tract-length) warping:
# a factor < 1 lowers the formants (longer tract -> male/deeper), > 1 raises them (brighter/child).
# Pitch is always the score's, so a character changes timbre, not the sung notes. This yields
# distinct male & female voices from the single available nnsvs voicebank (yoko, female).
CHARACTERS: dict[str, float] = {
    "yoko": 1.0,
    "female": 1.0,
    "soprano": 1.10,
    "child": 1.20,
    "alto": 0.93,
    "tenor": 0.86,
    "male": 0.82,
    "baritone": 0.78,
    "bass": 0.72,
}


def character_factor(name: str) -> float:
    """Formant-warp factor for a named voice character (default 1.0 = the base female voicebank)."""
    return CHARACTERS.get(name.lower(), 1.0)


# Reference spectral-envelope centroid (Hz), calibrated to the median across the bundled donor pool.
# A donor below this reads as darker/deeper (factor < 1), above as brighter (factor > 1). This is a
# timbre/brightness character, not a strict gender classifier (centroid also tracks the vowel).
_REF_CENTROID_HZ = 740.0


@functools.cache
def donor_formant_factor(wav_path: str) -> float:
    """Estimate a formant-warp factor from a donor recording (freesound/VocalSet), so its gross
    timbre/gender is carried onto the NNSVS singing. Pitch-independent (uses the WORLD envelope)."""
    import pyworld
    import soundfile as sf

    audio, sr = sf.read(wav_path)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size < sr // 10:
        return 1.0
    f0, t = pyworld.harvest(audio, sr, frame_period=10.0)
    sp = pyworld.cheaptrick(audio, f0, t, sr)
    voiced = sp[f0 > 0]
    if voiced.size == 0:
        return 1.0
    freqs = np.linspace(0.0, sr / 2.0, voiced.shape[1])
    centroid = float((voiced * freqs).sum() / max(voiced.sum(), 1e-9))
    return float(np.clip(centroid / _REF_CENTROID_HZ, 0.72, 1.20))


def _formant_warp(sp: np.ndarray, factor: float) -> np.ndarray:
    """Scale the spectral envelope's frequency axis by ``factor`` (vocal-tract-length warp)."""
    if abs(factor - 1.0) < 1e-3:
        return sp
    n_bins = sp.shape[1]
    src = np.arange(n_bins)
    query = np.clip(src / factor, 0, n_bins - 1)
    out = np.empty_like(sp)
    for f in range(sp.shape[0]):
        out[f] = np.interp(query, src, sp[f])
    return out
_NOTE_NAMES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
_NOTE_ALTER = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]


def to_kana(syllable: str | None) -> str:
    """Map a romaji-ish syllable to one hiragana mora (best effort); default 'ら' for empties."""
    if not syllable:
        return "ら"
    s = syllable.strip().lower()
    vowel = next((c for c in s if c in _VOWELS), "a")
    onset = s[: s.index(vowel)] if vowel in s else ""
    cons = _ROW.get(onset[-1], "") if onset else ""
    return _MORA.get(f"{cons}{vowel}") or _MORA.get(vowel, "ら")


def _xml_pitch(midi: int) -> str:
    step = _NOTE_NAMES[midi % 12]
    alter = _NOTE_ALTER[midi % 12]
    octave = midi // 12 - 1
    alt = f"<alter>{alter}</alter>" if alter else ""
    return f"<pitch><step>{step}</step>{alt}<octave>{octave}</octave></pitch>"


def _dur_type(divs: int, per_quarter: int) -> str:
    ratio = divs / per_quarter
    for r, name in [(4, "whole"), (2, "half"), (1, "quarter"), (0.5, "eighth"), (0.25, "16th")]:
        if ratio >= r * 0.75:
            return name
    return "16th"


def build_musicxml(notes: list[list[float]], syllables: list[str | None]) -> str:
    """Build a single-part MusicXML: one note per score note (kana lyric), rests for gaps."""
    per_quarter = 480  # divisions per quarter; tempo 120 => quarter = 0.5 s
    sec_to_div = per_quarter / 0.5
    body: list[str] = []

    def rest(dur_s: float) -> None:
        d = max(1, int(round(dur_s * sec_to_div)))
        body.append(
            f"<note><rest/><duration>{d}</duration><type>{_dur_type(d, per_quarter)}</type></note>"
        )

    prev_off = 0.0
    # Sinsy wants a leading rest.
    rest(max(0.2, notes[0][0]) if notes else 0.2)
    for (onset, offset, pitch), syl in zip(notes, syllables, strict=False):
        if onset - prev_off > 0.01:
            rest(onset - prev_off)
        d = max(1, int(round((offset - onset) * sec_to_div)))
        body.append(
            f"<note>{_xml_pitch(int(pitch))}<duration>{d}</duration>"
            f"<type>{_dur_type(d, per_quarter)}</type>"
            f"<lyric><syllabic>single</syllabic><text>{to_kana(syl)}</text></lyric></note>"
        )
        prev_off = offset
    rest(0.3)  # trailing rest

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="3.0"><part-list><score-part id="P1">'
        "<part-name>P</part-name></score-part></part-list>"
        '<part id="P1"><measure number="1">'
        f"<attributes><divisions>{per_quarter}</divisions><key><fifths>0</fifths></key>"
        "<time><beats>4</beats><beat-type>4</beat-type></time>"
        "<clef><sign>G</sign><line>2</line></clef></attributes>"
        '<direction><sound tempo="120"/></direction>'
        + "".join(body)
        + "</measure></part></score-partwise>"
    )


def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _tone(pitch: int, n: int, sr: int) -> np.ndarray:
    """A simple voiced tone at the score pitch — fallback when a segment can't be resynthesised."""
    t = np.arange(n) / sr
    f0 = _midi_to_hz(pitch)
    nyq = sr / 2
    sig = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 20) if k * f0 < nyq)
    sig = np.asarray(sig, dtype=np.float64)
    fade = min(int(0.005 * sr), n // 2)
    if fade > 0:
        ramp = np.linspace(0, 1, fade)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]
    return sig


def _voiced_runs(audio: np.ndarray, sr: int, hop_ms: float = 5.0) -> list[tuple[int, int]]:
    """Sample ranges of voiced (energetic) segments — one per sung syllable, rests excluded."""
    hop = max(1, int(sr * hop_ms / 1000.0))
    frame = 2 * hop
    n = audio.size
    if n == 0:
        return []
    rms = np.array(
        [np.sqrt(np.mean(audio[k : k + frame] ** 2)) for k in range(0, n, hop)], dtype=np.float64
    )
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0:
        return []
    voiced = rms > 0.12 * peak
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for k, v in enumerate(voiced):
        if v and start is None:
            start = k
        elif not v and start is not None:
            runs.append((start * hop, k * hop))
            start = None
    if start is not None:
        runs.append((start * hop, n))
    return [(a, b) for a, b in runs if b - a >= frame]


def align_and_pitchlock(
    audio: np.ndarray,
    notes: list[list[float]],
    sr: int,
    frame_period: float = 5.0,
    formant_factor: float = 1.0,
) -> np.ndarray:
    """Re-place each sung syllable on the score grid at the exact score pitch (WORLD resynthesis).

    Keeps NNSVS's timbre + consonants (the segment's spectral envelope / aperiodicity) but imposes
    the score's f0 (flat at the MIDI pitch) and onset/offset, so onsets/pitch are exact by
    construction and the sample passes the validator. The expressive (non-locked) render is what
    lands in rejected/ today; this is the corpus-valid variant.
    """
    import pyworld

    runs = _voiced_runs(audio, sr)
    if not runs or not notes:
        return audio
    total = max(off for _, off, _ in notes)
    out = np.zeros(int(round(total * sr)) + sr, dtype=np.float64)
    n_runs, n_notes = len(runs), len(notes)
    for i, (onset, offset, pitch) in enumerate(notes):
        dur = offset - onset
        if dur <= 0:
            continue
        a, b = runs[min(n_runs - 1, round(i * n_runs / n_notes))]  # proportional note->segment map
        seg = audio[a:b].astype(np.float64)
        n_samp = int(round(dur * sr))
        note_audio: np.ndarray | None = None
        if seg.size >= 2 * sr // 100:  # long enough to analyse
            f0, t = pyworld.harvest(seg, sr, frame_period=frame_period)
            if np.any(f0 > 0):  # voiced segment: transplant its timbre at the score pitch
                sp = _formant_warp(pyworld.cheaptrick(seg, f0, t, sr), formant_factor)
                ap = pyworld.d4c(seg, f0, t, sr)
                n_tgt = max(1, int(round(dur * 1000.0 / frame_period)))
                idx = np.clip(
                    np.round(np.linspace(0, sp.shape[0] - 1, n_tgt)).astype(int), 0, sp.shape[0] - 1
                )
                # A tiny bit of vibrato around the score pitch (fading in after the attack) so the
                # audio isn't dead-flat — closer to real singing for the transcriber to generalise.
                # The LABEL stays the exact score pitch: vibrato is centred on it and its ~18-cent
                # peak stays under the validator's 25-cent f0 tolerance, so coverage isn't hurt.
                tt = np.arange(n_tgt) * (frame_period / 1000.0)
                vib_gain = np.clip(tt / 0.12, 0.0, 1.0)  # fade in over ~120 ms
                vib_cents = 18.0 * vib_gain * np.sin(2.0 * np.pi * 5.5 * tt)
                f0_tgt = _midi_to_hz(int(pitch)) * (2.0 ** (vib_cents / 1200.0))
                note_audio = pyworld.synthesize(f0_tgt, sp[idx], ap[idx], sr, frame_period)
        if note_audio is None or note_audio.size == 0 or float(np.max(np.abs(note_audio))) < 1e-4:
            note_audio = _tone(int(pitch), n_samp, sr)  # fallback: in-tune tone at the score pitch
        # Snap energy onset/offset to the score grid: force the exact note length and apply a sharp
        # attack/release so the validator's onset/offset detection fires at (onset, offset) even for
        # short, dense notes (real-singing phrases) where WORLD's soft attack would otherwise lag.
        if note_audio.size >= n_samp:
            note_audio = note_audio[:n_samp]
        else:
            note_audio = np.pad(note_audio, (0, n_samp - note_audio.size))
        atk = min(int(0.003 * sr), n_samp // 4)
        rel = min(int(0.012 * sr), n_samp // 4)
        if atk > 0:
            note_audio[:atk] *= np.linspace(0.0, 1.0, atk)
        if rel > 0:
            note_audio[-rel:] *= np.linspace(1.0, 0.0, rel)
        start = int(round(onset * sr))
        end = min(start + note_audio.size, out.size)
        out[start:end] += note_audio[: end - start]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out *= 0.9 / peak
    return out[: int(round(total * sr)) + 1].astype(np.float32)


@functools.cache
def _load_engine(model_ref: str):
    import torch
    from nnsvs.pretrained import retrieve_pretrained_model
    from nnsvs.svs import SPSVS

    model_dir = retrieve_pretrained_model(model_ref)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SPSVS(model_dir, device=device)


def render_nnsvs(
    notes: list[list[float]],
    syllables: list[str | None],
    sr: int,
    model_ref: str = DEFAULT_MODEL,
    pitch_lock: bool = True,
    formant_factor: float = 1.0,
) -> np.ndarray:
    """Render natural singing for ``notes`` + ``syllables`` with NNSVS, resampled to ``sr``.

    With ``pitch_lock`` (default), each sung syllable is re-placed on the score grid at the exact
    score pitch/timing via WORLD resynthesis (corpus-valid: passes the onset/pitch gates while
    keeping NNSVS timbre). Set it False for the raw, expressive render.
    """
    import pysinsy
    from nnmnkwii.io import hts
    from scipy.signal import resample_poly

    engine = _load_engine(model_ref)
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "score.xml"
        xml.write_text(build_musicxml(notes, syllables), encoding="utf-8")
        sinsy = pysinsy.sinsy.Sinsy()
        sinsy.setLanguages("j", pysinsy.get_default_dic_dir())
        sinsy.loadScoreFromMusicXML(str(xml))
        lines = sinsy.createLabelData(False, 1, 1).getData()
        sinsy.clearScore()
        lab = Path(tmp) / "score.lab"
        lab.write_text("\n".join(lines) + "\n", encoding="utf-8")
        labels = hts.load(str(lab))

    wav, model_sr = engine.svs(labels, vocoder_type="world")
    wav = np.asarray(wav, dtype=np.float64)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav / peak * 0.9
    if model_sr != sr:
        wav = resample_poly(wav, sr, model_sr)
    if pitch_lock:
        wav = align_and_pitchlock(wav, notes, sr, formant_factor=formant_factor)
    return wav.astype(np.float32)
