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
        from rvc_python.infer import RVCInference
    except Exception as exc:  # pragma: no cover - exercised only without the toolkit installed
        print(json.dumps({"ok": False, "error": f"rvc-python not importable: {exc}"}))
        return 4

    device = _select_device(str(req.get("device", "auto")))
    out_wav = req["out_wav"]
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

    rvc = RVCInference(device=device)
    rvc.load_model(model_ref)
    rvc.set_params(f0up_key=int(req.get("f0up_key", 0)))  # 0 -> keep the input (score) pitch
    rvc.infer_file(req["in_wav"], out_wav)

    print(json.dumps({"ok": True, "out_wav": out_wav, "device": device, "backend": "rvc"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
