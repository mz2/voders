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

# DiffSinger SVS archives (GitHub releases, unpacked rather than placed file-for-file). The TIGER
# English voicebank is CC BY-NC-ND 4.0 (NON-COMMERCIAL) — gate it behind your corpus license policy.
_TIGER_PACK_URL = "https://github.com/spicytigermeat/tiger_diffsinger/releases/download/v106/TIGER_DS_v106_PACK.zip"
_NSF_VOCODER_URL = "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep"


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


def _extract_zip(archive: Path, dest: Path, *, strip_to: str | None = None) -> None:
    """Unzip ``archive`` into ``dest``. If ``strip_to`` is given, only members under the first path
    component containing it are extracted, flattened to ``dest`` (handles the nested TIGER layout)."""
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if strip_to is not None:
                if strip_to not in member:
                    continue
                rel = member.split(strip_to, 1)[1].lstrip("/")
            else:
                rel = member
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                out.write(src.read())


def _download_diffsinger() -> None:
    """Fetch + unpack the TIGER voicebank and NSF-HiFiGAN vocoder into models/diffsinger/."""
    import tempfile

    root = MODELS_ROOT / "diffsinger"
    tiger_dir, voc_dir = root / "tiger", root / "nsf_hifigan"
    if (tiger_dir / "dsacoustic" / "acoustic.onnx").exists() and any(voc_dir.glob("*.onnx")):
        print(f"  exists, skipping: {tiger_dir} and {voc_dir}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pack = tmp_path / "tiger_pack.zip"
        _download(_TIGER_PACK_URL, pack)
        # pack -> .../Voice Library/TIGER_DS_v106.zip (the actual voicebank) -> tiger/
        _extract_zip(pack, tmp_path / "pack")
        inner = next((tmp_path / "pack").rglob("TIGER_DS_v*.zip"))
        _extract_zip(inner, tiger_dir)
        voc = tmp_path / "vocoder.zip"
        _download(_NSF_VOCODER_URL, voc)  # .oudep is a renamed zip
        _extract_zip(voc, voc_dir)
    print(f"  done: {tiger_dir} ({sum(f.stat().st_size for f in tiger_dir.rglob('*')) / 1e6:.0f} MB)")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    which = args or ["rvc"]
    for group in which:
        if group == "diffsinger":
            print(f"== diffsinger -> {MODELS_ROOT / 'diffsinger'} ==")
            _download_diffsinger()
            continue
        if group not in MODELS:
            print(f"unknown model group {group!r}; known: {sorted([*MODELS, 'diffsinger'])}")
            return 2
        print(f"== {group} -> {MODELS_ROOT} ==")
        for relpath, url in MODELS[group]:
            _download(url, MODELS_ROOT / relpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
