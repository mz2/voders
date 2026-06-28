"""DiffSinger render worker — a CLI invoked by the 3.14 core across a process boundary.

Protocol (mirrors the nnsvs backend so the core's backend_bridge is identical):
  argv[1] = path to a JSON request: {"notes": [[onset_s, offset_s, pitch_midi], ...], "sr": 22050,
            "seed": int, "out_wav": path, "model_ref": str, "lyrics": [...], "phonemes": [...]}
  On success: writes the WAV to out_wav and prints {"ok": true, "n_samples": int,
              "backend": "diffsinger", "articulated": bool, "speaker": str} to stdout; exit 0.

``model_ref`` selects the voicebank + voice-colour speaker embedding:
  "diffsinger" / ""            -> default voicebank, default speaker
  "diffsinger:<speaker>"       -> default voicebank, named voice colour (e.g. tiger_glam)
  "diffsinger:<bank>:<speaker>"-> named bank dir under models/diffsinger/ + speaker

Score pitch/timing are exact by construction (f0 + durations are model inputs), so the render is
corpus-valid without any pitch-relock — see diffsinger_engine for the tensor contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

from voders_diffsinger_backend.diffsinger_engine import (
    DEFAULT_MODEL_DIR,
    DEFAULT_SPEAKER,
    _MODELS,
    render_diffsinger,
)


def _resolve_model(model_ref: str) -> tuple[Path, str]:
    """Map a voice ``model_ref`` to a (voicebank dir, speaker) pair."""
    ref = (model_ref or "").strip()
    if not ref or ref == "diffsinger":
        return DEFAULT_MODEL_DIR, DEFAULT_SPEAKER
    parts = ref.split(":")
    if parts[0] == "diffsinger":
        parts = parts[1:]
    if len(parts) == 1:
        return DEFAULT_MODEL_DIR, parts[0]
    if len(parts) >= 2:
        return _MODELS / parts[0], parts[1]
    return DEFAULT_MODEL_DIR, DEFAULT_SPEAKER


def _handle(req: dict) -> dict:
    sr = int(req.get("sr", 22050))
    notes = req["notes"]
    if not notes:
        return {"ok": False, "error": "diffsinger requires at least one note"}
    lyrics = req.get("lyrics")
    phonemes = req.get("phonemes")
    model_dir, speaker = _resolve_model(str(req.get("model_ref", "")))

    audio = render_diffsinger(
        notes,
        sr,
        phonemes=phonemes,
        lyrics=lyrics,
        model_dir=model_dir,
        speaker=speaker,
    )

    out_wav = req["out_wav"]
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, audio, sr, subtype="FLOAT")
    return {
        "ok": True,
        "n_samples": int(audio.size),
        "backend": "diffsinger",
        "articulated": bool(phonemes) or bool(lyrics),
        "speaker": speaker,
    }


def _serve() -> int:
    """Persistent mode: one JSON request per stdin line, one JSON response per stdout line.

    The ONNX sessions are process-cached (functools.cache in the engine), so a long batch render
    loads the acoustic + vocoder models once and reuses them — the core's backend_bridge spawns one
    of these with VODERS_SVS_PERSISTENT=1.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = _handle(json.loads(line))
        except Exception as exc:  # noqa: BLE001 — report per-request, keep the worker alive
            resp = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "--serve":
        return _serve()
    if not args:
        print(json.dumps({"ok": False, "error": "missing request path"}))
        return 2
    print(json.dumps(_handle(json.loads(Path(args[0]).read_text(encoding="utf-8")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
