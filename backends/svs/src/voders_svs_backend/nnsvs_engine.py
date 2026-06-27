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
) -> np.ndarray:
    """Render natural singing for ``notes`` + ``syllables`` with NNSVS, resampled to ``sr``."""
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
    return wav.astype(np.float32)
