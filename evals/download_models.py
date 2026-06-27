"""Download neural backend model weights (run via uv: `uv run python evals/download_models.py`).

Fetches the **base/foundation** models the RVC voice-conversion toolkit needs — a HuBERT content
encoder and the RMVPE pitch extractor. These are general feature extractors, not a cloned
individual's voice, so they carry no consent question. A *target singer* model (an RVC `.pth`) is a
separate, consent-gated asset the operator must supply via a voice's ``model_ref`` (FR-011).

Weights land under ``models/`` (git-ignored — large binaries are never committed; the constitution
allows only small fixtures via LFS).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Each entry is (destination path under models/, download URL).
_HF_RVC_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
_HF_VCTK = "https://huggingface.co/Nekochu/RVC-VCTK_Voice-sample/resolve/main"

MODELS = {
    # RVC base/foundation models (feature extractors, not a cloned voice).
    "rvc": [
        ("rvc/hubert_base.pt", f"{_HF_RVC_BASE}/hubert_base.pt"),
        ("rvc/rmvpe.pt", f"{_HF_RVC_BASE}/rmvpe.pt"),
    ],
    # A consented target voice: VCTK speaker p231 (female). The model repo is Apache-2.0 and VCTK
    # is CC BY 4.0 (speakers consented to open release) — a defensible, license-clean target.
    "vctk-p231": [
        ("rvc/voices/Fp231rmvpe.pth", f"{_HF_VCTK}/F/p231/rmvpe/Fp231rmvpe.pth"),
        (
            "rvc/voices/Fp231rmvpe.index",
            f"{_HF_VCTK}/F/p231/rmvpe/added_IVF1216_Flat_nprobe_1_Fp231rmvpe_v2.index",
        ),
    ],
}

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists, skipping: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  downloading {dest.name} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "voders-model-fetch"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:  # noqa: S310 - pinned https URL
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r    {read / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    print(f"  done: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    which = args or ["rvc"]
    for group in which:
        if group not in MODELS:
            print(f"unknown model group {group!r}; known: {sorted(MODELS)}")
            return 2
        print(f"== {group} -> {MODELS_ROOT} ==")
        for relpath, url in MODELS[group]:
            _download(url, MODELS_ROOT / relpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
