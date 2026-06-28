"""SoulX-Singer SVS render worker — invoked by the 3.14 core across a process boundary.

SoulX-Singer (Apache-2.0) is a zero-shot singing synthesiser: it sings a *target* score (per-note
duration / MIDI / phoneme) in the timbre of a *prompt* reference clip. This worker converts the
core's request (notes + per-note syllables + language) into SoulX's target metadata, runs inference
with control="score" (MIDI-conditioned, so onsets/pitch follow the score), and writes a WAV.

Languages: SoulX covers English + Mandarin (+ Cantonese). The prompt defaults to SoulX's bundled
English/Mandarin reference clip per language. The heavy 704M model loads once per process, so the
``--serve`` loop (model-resident) is how batch renders stay fast.

Protocol mirrors the other backends:
  argv[1] = path to a JSON request, OR ``--serve`` for one request/response per stdin/stdout line.
  request = {"notes": [[on,off,midi]...], "lyrics": [syll|null...], "language": "en-us"|"cmn",
             "out_wav": path, "sr": 24000, "device": "cuda"}
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# SoulX's build_model / inference print progress to stdout, which would corrupt the JSON
# protocol. Reserve the real stdout for our JSON; everything else goes to stderr.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

_BACKEND_DIR = Path(__file__).resolve().parents[2]  # backends/soulx
_REPO = _BACKEND_DIR / "SoulX-Singer"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_MODEL_PATH = _REPO / "pretrained_models" / "SoulX-Singer" / "model.pt"
_CONFIG_PATH = _REPO / "soulxsinger" / "config" / "soulxsinger.yaml"
_PHONESET = _REPO / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
# Bundled reference clips (timbre source) + their metadata, by SoulX language tag.
_PROMPTS = {
    "English": (_REPO / "example/audio/en_prompt.mp3", _REPO / "example/audio/en_prompt.json"),
    "Mandarin": (_REPO / "example/audio/zh_prompt.mp3", _REPO / "example/audio/zh_prompt.json"),
}
# Our espeak-ish language codes -> SoulX language tag (only en/zh/yue are supported by SoulX).
_LANG = {"en-us": "English", "en": "English", "cmn": "Mandarin", "zh": "Mandarin"}


def _g2p(words: list[str], language: str) -> list[str]:
    """Per-note phoneme tokens (e.g. ``en_HH-UW1``) via SoulX's own g2p."""
    from preprocess.tools.g2p import g2p_transform

    return g2p_transform(words, language)


def build_target_meta(
    notes: list[list[float]], lyrics: list[str | None], language: str, gap_s: float = 0.12
) -> dict:
    """Convert (notes, per-note syllables) into one SoulX target-metadata segment.

    A rest longer than ``gap_s`` between notes becomes a ``<SP>`` (note_pitch 0). note_type is 1 for
    every note (each is an independent one-syllable-per-note onset) and for ``<SP>``. f0 is left
    empty: control="score" drives pitch from note_pitch (MIDI).
    """
    words: list[str] = []
    durs: list[float] = []
    pitches: list[int] = []
    prev_off: float | None = None
    for i, (onset, offset, pitch) in enumerate(notes):
        syl = lyrics[i] if i < len(lyrics) else None
        if prev_off is not None and onset - prev_off > gap_s:
            words.append("<SP>")
            durs.append(round(onset - prev_off, 3))
            pitches.append(0)
        words.append(str(syl) if syl else "la")
        durs.append(round(offset - onset, 3))
        pitches.append(int(pitch))
        prev_off = offset

    # g2p only the real syllables; <SP> stays literal.
    real = [w for w in words if w != "<SP>"]
    phon_real = iter(_g2p(real, language))
    phonemes = [w if w == "<SP>" else next(phon_real) for w in words]
    total_ms = int(round(sum(durs) * 1000))
    return {
        "index": "vocal_0",
        "language": language,
        "time": [0, total_ms],
        "duration": " ".join(f"{d:.2f}" for d in durs),
        "text": " ".join(words),
        "phoneme": " ".join(phonemes),
        "note_pitch": " ".join(str(p) for p in pitches),
        "note_type": " ".join("1" for _ in words),
        "f0": "",
    }


@functools.cache
def _engine(device: str):
    """Load the SoulX model, config, data processor, and per-language prompt once."""
    import torch  # noqa: F401
    from cli.inference import build_model
    from soulxsinger.utils.data_processor import DataProcessor
    from soulxsinger.utils.file_utils import load_config

    config = load_config(_CONFIG_PATH)
    model = build_model(str(_MODEL_PATH), config, device=device, use_fp16=True)
    dp = DataProcessor(
        hop_size=config.audio.hop_size,
        sample_rate=config.audio.sample_rate,
        phoneset_path=str(_PHONESET),
        device=device,
    )
    return model, config, dp


@functools.cache
def _prompt(device: str, language: str):
    _, _, dp = _engine(device)
    wav, meta = _PROMPTS[language]
    prompt_meta = json.loads(Path(meta).read_text(encoding="utf-8"))[0]
    return dp.process(prompt_meta, str(wav))


def _handle(req: dict) -> dict:
    import torch

    device = str(req.get("device", "cuda"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    language = _LANG.get(str(req.get("language", "en-us")), "English")
    model, config, dp = _engine(device)
    prompt_data = _prompt(device, language)

    meta = build_target_meta(req["notes"], req.get("lyrics") or [], language)
    target_data = dp.process(meta, None)
    with torch.no_grad():
        audio = model.infer(
            {"prompt": prompt_data, "target": target_data},
            auto_shift=False,  # keep the exact score MIDI so onsets/pitch match the labels
            pitch_shift=0,
            n_steps=config.infer.n_steps,
            cfg=config.infer.cfg,
            control="score",
            use_fp16=True,
        )
    audio = np.asarray(audio.squeeze().cpu().numpy(), dtype=np.float32)
    out_wav = req["out_wav"]
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, audio, int(config.audio.sample_rate), subtype="FLOAT")
    return {"ok": True, "n_samples": int(audio.size), "backend": "soulx", "language": language}


def _serve() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = _handle(json.loads(line))
        except Exception as exc:  # noqa: BLE001 — per-request error, keep the worker alive
            resp = {"ok": False, "error": str(exc)}
        _REAL_STDOUT.write(json.dumps(resp) + "\n")
        _REAL_STDOUT.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "--serve":
        return _serve()
    if not args:
        _REAL_STDOUT.write(json.dumps({"ok": False, "error": "missing request path"}) + "\n")
        return 2
    resp = _handle(json.loads(Path(args[0]).read_text(encoding="utf-8")))
    _REAL_STDOUT.write(json.dumps(resp) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
