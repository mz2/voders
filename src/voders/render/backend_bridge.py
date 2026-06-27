"""Bridge to out-of-process render backends managed by their own uv projects.

Some lane toolkits (e.g. NNSVS) pin dependencies that conflict with the 3.14 core, so they live in
standalone uv projects under ``backends/`` with their own ``pyproject.toml`` / ``.python-version`` /
``uv.lock``. The core invokes them with ``uv run --project <dir>`` — a fresh interpreter with the
backend's own dependency chain — passing a small JSON request and reading back a WAV. This keeps the
two incompatible dependency chains in separate processes (the reason the project uses uv).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from voders.audio import read_wav
from voders.constants import SAMPLE_RATE
from voders.scores.models import Score


def backends_root() -> Path:
    """Directory holding the standalone backend projects (override with ``VODERS_BACKENDS_DIR``)."""
    env = os.environ.get("VODERS_BACKENDS_DIR")
    if env:
        return Path(env)
    # src/voders/render/backend_bridge.py -> repo root is three parents up from the package dir.
    return Path(__file__).resolve().parents[3] / "backends"


def backend_available(name: str) -> bool:
    """True when the named backend project exists and has been synced (has a .venv)."""
    proj = backends_root() / name
    return (proj / "pyproject.toml").exists() and (proj / ".venv").exists()


def render_via_backend(
    name: str,
    module: str,
    score: Score,
    seed: int,
    *,
    model_ref: str = "",
    mode: str = "force_score_f0",
    sr: int = SAMPLE_RATE,
    timeout_s: float = 300.0,
) -> np.ndarray:
    """Render ``score`` in the backend project ``name`` and return the audio.

    Runs ``uv run --project backends/<name> python -m <module> <request.json>``; the worker writes
    a WAV which is read back here. Raises RuntimeError if the backend project is missing or the
    worker fails.
    """
    proj = backends_root() / name
    if not (proj / "pyproject.toml").exists():
        raise RuntimeError(
            f"backend project {name!r} not found at {proj}; run `uv sync` in that directory"
        )

    with tempfile.TemporaryDirectory() as tmp:
        req_path = Path(tmp) / "request.json"
        out_wav = Path(tmp) / "out.wav"
        request = {
            "notes": [[n.onset_s, n.offset_s, n.pitch_midi] for n in score.notes],
            "sr": sr,
            "seed": int(seed),
            "out_wav": str(out_wav),
            "model_ref": model_ref,
            "mode": mode,
        }
        req_path.write_text(json.dumps(request), encoding="utf-8")

        proc = subprocess.run(
            ["uv", "run", "--project", str(proj), "python", "-m", module, str(req_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"backend {name!r} failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}"
            )
        if not out_wav.exists():
            raise RuntimeError(f"backend {name!r} produced no audio at {out_wav}")
        audio, _ = read_wav(out_wav)
        return audio
