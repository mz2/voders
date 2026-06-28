"""Bridge to out-of-process render backends managed by their own uv projects.

Some lane toolkits (e.g. NNSVS) pin dependencies that conflict with the 3.14 core, so they live in
standalone uv projects under ``backends/`` with their own ``pyproject.toml`` / ``.python-version`` /
``uv.lock``. The core invokes them with ``uv run --project <dir>`` — a fresh interpreter with the
backend's own dependency chain — passing a small JSON request and reading back a WAV. This keeps the
two incompatible dependency chains in separate processes (the reason the project uses uv).
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from voders.audio import read_wav, write_wav
from voders.constants import SAMPLE_RATE
from voders.scores.models import Score

# --- Persistent (model-resident) backend workers -------------------------------------------------
# By default each render spawns a fresh worker process (model reloads every sample), which dominates
# batch-render wall time and — for GPU backends run many shards wide — multiplies GPU memory until
# it OOMs. With VODERS_SVS_PERSISTENT=1 the core keeps ONE long-lived worker per (backend, module)
# in `--serve` mode: the model loads once and is reused for every request, so a big render pays the
# load cost once and holds a single model in GPU memory.
_PERSISTENT: dict[tuple[str, str], _PersistentBackend] = {}
_PERSISTENT_LOCK = threading.Lock()


class _PersistentBackend:
    """A long-lived ``--serve`` worker: one resident model, streamed JSON requests/responses."""

    def __init__(self, name: str, module: str) -> None:
        proj = backends_root() / name
        if not (proj / "pyproject.toml").exists():
            raise RuntimeError(f"backend project {name!r} not found at {proj}")
        self.name = name
        self.proc = subprocess.Popen(
            ["uv", "run", "--project", str(proj), "python", "-m", module, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, req: dict) -> dict:
        if self.proc.poll() is not None or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError(f"persistent backend {self.name!r} is not running")
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"persistent backend {self.name!r} closed its output")
        return json.loads(line)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass


def _persistent_backend(name: str, module: str) -> _PersistentBackend:
    with _PERSISTENT_LOCK:
        key = (name, module)
        backend = _PERSISTENT.get(key)
        if backend is None or backend.proc.poll() is not None:
            backend = _PersistentBackend(name, module)
            _PERSISTENT[key] = backend
            atexit.register(backend.close)
        return backend


def _persistent_enabled() -> bool:
    return os.environ.get("VODERS_SVS_PERSISTENT") == "1"


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
    lyrics: list[str | None] | None = None,
    phonemes: list[dict] | None = None,
    expr_scale: float | None = None,
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
        # Optional per-note syllables + core-side G2P phonemes for an articulating SVS backend
        # (FR-006, Decision L3); absent => the backend sings an open vowel.
        if lyrics is not None and any(s for s in lyrics):
            request["lyrics"] = list(lyrics)
        if phonemes:
            request["phonemes"] = phonemes
        # NNSVS pitch-lock vibrato retention; omitted => the backend uses its own default.
        if expr_scale is not None:
            request["expr_scale"] = float(expr_scale)

        if _persistent_enabled():
            # Reuse one model-resident worker (no per-sample reload, single GPU model). The worker
            # writes out_wav; we read it back here exactly as in the one-shot path.
            resp = _persistent_backend(name, module).request(request)
            if not resp.get("ok"):
                raise RuntimeError(f"backend {name!r}: {resp.get('error', 'render failed')}")
        else:
            req_path.write_text(json.dumps(request), encoding="utf-8")
            proc = subprocess.run(
                ["uv", "run", "--project", str(proj), "python", "-m", module, str(req_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if proc.returncode != 0:
                tail = proc.stderr.strip()[-500:]
                raise RuntimeError(f"backend {name!r} failed (exit {proc.returncode}): {tail}")
        if not out_wav.exists():
            raise RuntimeError(f"backend {name!r} produced no audio at {out_wav}")
        audio, _ = read_wav(out_wav)
        return audio


def _run_worker(name: str, module: str, request: dict, files: list[Path], timeout_s: float) -> dict:
    proj = backends_root() / name
    if not (proj / "pyproject.toml").exists():
        raise RuntimeError(f"backend project {name!r} not found at {proj}; run `uv sync` there")
    proc = subprocess.run(
        ["uv", "run", "--project", str(proj), "python", "-m", module, *[str(f) for f in files]],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"backend {name!r} failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def generate_lyrics_via_backend(
    theme: str,
    score: Score,
    seed: int,
    *,
    model_ref: str = "",
    timeout_s: float = 600.0,
) -> str:
    """Run the out-of-process lyrics model backend and return generated lyric text (FR-011).

    The core segments the returned text into one-syllable-per-note (FR-019). Raises RuntimeError if
    the backend project is missing or the worker fails (mirrors the other backends).
    """
    with tempfile.TemporaryDirectory() as tmp:
        req_path = Path(tmp) / "request.json"
        request = {
            "theme": theme,
            "score_id": score.score_id,
            "notes": [[n.onset_s, n.offset_s, n.pitch_midi] for n in score.notes],
            "seed": int(seed),
            "model_ref": model_ref,
        }
        req_path.write_text(json.dumps(request), encoding="utf-8")
        result = _run_worker("lyrics", "voders_lyrics_backend.worker", {}, [req_path], timeout_s)
        if "text" in result:
            return str(result["text"])
        if "syllables" in result:
            return " ".join(str(s) for s in result["syllables"] if s)
        raise RuntimeError("lyrics backend returned neither 'text' nor 'syllables'")


def convert_via_backend(
    name: str,
    module: str,
    audio: np.ndarray,
    model_ref: str,
    *,
    device: str = "auto",
    sr: int = SAMPLE_RATE,
    timeout_s: float = 600.0,
    params: dict | None = None,
) -> np.ndarray:
    """Send already-rendered audio to a conversion backend and return the converted audio.

    Used by the voice-conversion lane's neural backends (e.g. RVC, Seed-VC): the core renders
    score-aligned audio in-process, the backend changes only the timbre, and the score f0 (labels)
    is preserved. ``params`` carries backend-specific options (e.g. ``diffusion_steps``).
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_wav = Path(tmp) / "in.wav"
        out_wav = Path(tmp) / "out.wav"
        req_path = Path(tmp) / "request.json"
        write_wav(in_wav, audio, sr)
        request = {
            "in_wav": str(in_wav),
            "out_wav": str(out_wav),
            "model_ref": model_ref,
            "device": device,
            "f0up_key": 0,
        }
        request.update(params or {})
        req_path.write_text(json.dumps(request), encoding="utf-8")
        result = _run_worker(name, module, {}, [req_path], timeout_s)
        if not result.get("ok"):
            raise RuntimeError(f"backend {name!r}: {result.get('error', 'conversion failed')}")
        converted, csr = read_wav(out_wav)
        if csr != sr:
            # Conversion backends (RVC) emit at their own target rate; bring it back to 22,050 Hz.
            import librosa

            converted = librosa.resample(converted, orig_sr=csr, target_sr=sr)
        return converted
