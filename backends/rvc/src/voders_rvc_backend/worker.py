"""RVC voice-conversion worker — invoked by the 3.14 core across a process boundary.

Protocol:
  argv[1] = path to a JSON request: {"in_wav": path, "out_wav": path, "model_ref": path-to-.pth,
                                      "device": "auto|cuda|cpu", "f0up_key": 0}
  On success: writes the converted WAV to out_wav and prints {"ok": true, ...}; exit 0.

RVC keeps the input audio's pitch contour (we feed it the score-aligned WORLD render), so the
score f0 — and the labels — are preserved. A *consented* target voice model (`model_ref`, an RVC
`.pth`) is required: there is no default voice (FR-011).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _patch_torch_load() -> None:
    """Default torch.load to weights_only=False.

    PyTorch >=2.6 defaults weights_only=True, which rejects the older RVC/fairseq checkpoints
    (they pickle non-tensor objects). The models loaded here are trusted, locally-downloaded files
    (the official RVC base models and the Apache-2.0 VCTK target voice), so this is safe.
    """
    import torch

    if getattr(torch.load, "_voders_patched", False):
        return
    _orig = torch.load

    def load(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    load._voders_patched = True  # type: ignore[attr-defined]
    torch.load = load  # type: ignore[assignment]


def _extract_audio(result: object):  # noqa: ANN202 - returns np.ndarray | None
    """Pull the 1-D audio array out of RVC's vc_single return (an ndarray, or a nested tuple)."""
    import numpy as np

    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, tuple):
        for item in reversed(result):
            found = _extract_audio(item)
            if found is not None:
                return found
    return None


def _select_device(pref: str) -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(json.dumps({"ok": False, "error": "missing request path"}))
        return 2
    req = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    model_ref = req.get("model_ref") or ""
    if not model_ref or not Path(model_ref).exists():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "RVC requires a consented target voice model (.pth) via model_ref; "
                    f"got {model_ref!r}",
                }
            )
        )
        return 3

    try:
        _patch_torch_load()
        from rvc_python.infer import RVCInference
    except Exception as exc:  # pragma: no cover - exercised only without the toolkit installed
        print(json.dumps({"ok": False, "error": f"rvc-python not importable: {exc}"}))
        return 4

    device = _select_device(str(req.get("device", "auto")))
    out_wav = req["out_wav"]
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

    # Pair the model with its retrieval index (same stem, .index) when present.
    index_path = req.get("index_path") or ""
    if not index_path:
        candidate = Path(model_ref).with_suffix(".index")
        index_path = str(candidate) if candidate.exists() else ""

    rvc = RVCInference(device=device)
    rvc.load_model(model_ref, index_path=index_path)
    # f0up_key=0 keeps the input (score-aligned) pitch; rmvpe extracts f0 from the input audio.
    rvc.set_params(f0method="rmvpe", f0up_key=int(req.get("f0up_key", 0)))

    # rvc-python 0.0.6's infer_file passes vc_single's tuple straight to wavfile.write (a bug), so
    # call vc_single directly and unwrap the audio array ourselves.
    import soundfile as sf

    file_index = rvc.models[rvc.current_model].get("index", "") or index_path
    result = rvc.vc.vc_single(
        sid=0,
        input_audio_path=req["in_wav"],
        f0_up_key=rvc.f0up_key,
        f0_method=rvc.f0method,
        file_index=file_index,
        index_rate=rvc.index_rate,
        filter_radius=rvc.filter_radius,
        resample_sr=rvc.resample_sr,
        rms_mix_rate=rvc.rms_mix_rate,
        protect=rvc.protect,
        f0_file="",
        file_index2="",
    )
    audio = _extract_audio(result)
    if audio is None:
        print(json.dumps({"ok": False, "error": f"could not extract audio from {type(result)}"}))
        return 5
    sf.write(out_wav, audio, int(rvc.vc.tgt_sr))

    print(
        json.dumps(
            {
                "ok": True,
                "out_wav": out_wav,
                "device": device,
                "index": bool(index_path),
                "backend": "rvc",
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
