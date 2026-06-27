"""Seed-VC zero-shot voice-conversion worker — invoked by the 3.14 core across a process boundary.

Seed-VC is a git repo (not a PyPI package), so this worker clones it once and runs its
``inference.py`` in this backend's Python 3.10 environment. It is *zero-shot*: the target voice is
a reference audio clip (``model_ref`` — e.g. one of the consented VocalSet/VCTK donor wavs), not a
trained per-voice model. ``--f0-condition True`` keeps the source pitch so the labels are preserved.

Protocol:
  argv[1] = JSON request {"in_wav", "out_wav", "model_ref" (reference wav), "device",
                          "diffusion_steps"}
  On success: writes out_wav and prints {"ok": true, ...}; exit 0.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_URL = "https://github.com/Plachtaa/seed-vc"
# backends/seedvc/src/voders_seedvc_backend/worker.py -> parents[2] == backends/seedvc
_REPO_DIR = Path(__file__).resolve().parents[2] / "seed-vc"  # backends/seedvc/seed-vc


def _ensure_repo() -> Path:
    if not (_REPO_DIR / "inference.py").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, str(_REPO_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
    return _REPO_DIR


def _select_device(pref: str) -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(json.dumps({"ok": False, "error": "missing request path"}))
        return 2
    req = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    ref = req.get("model_ref") or ""
    if not ref or not Path(ref).exists():
        print(
            json.dumps(
                {"ok": False, "error": f"seedvc needs a reference wav (model_ref); got {ref!r}"}
            )
        )
        return 3

    repo = _ensure_repo()
    device = _select_device(str(req.get("device", "auto")))
    out_wav = Path(req["out_wav"]).resolve()
    out_dir = out_wav.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # inference.py runs with cwd=repo (for its relative imports), so all I/O paths must be absolute.
    src_abs = str(Path(req["in_wav"]).resolve())
    ref_abs = str(Path(ref).resolve())

    cmd = [
        sys.executable,
        str(repo / "inference.py"),
        "--source",
        src_abs,
        "--target",
        ref_abs,
        "--output",
        str(out_dir),
        "--diffusion-steps",
        str(req.get("diffusion_steps", 25)),
        "--length-adjust",
        "1.0",
        "--inference-cfg-rate",
        "0.7",
        "--f0-condition",
        "True",  # keep the source (score-aligned) pitch
        "--fp16",
        "True" if device == "cuda" else "False",
    ]
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(json.dumps({"ok": False, "error": f"seed-vc inference failed: {proc.stderr[-500:]}"}))
        return 4

    # Seed-VC writes a wav into the output dir; pick the newest and move it to out_wav.
    produced = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    produced = [p for p in produced if p != out_wav]
    if not produced:
        print(json.dumps({"ok": False, "error": "seed-vc produced no wav"}))
        return 5
    shutil.move(str(produced[-1]), str(out_wav))
    print(json.dumps({"ok": True, "out_wav": str(out_wav), "device": device, "backend": "seedvc"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
