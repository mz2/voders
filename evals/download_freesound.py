"""Fetch consented donor vowels from Freesound (https://freesound.org).

Run via uv with the cpu extra and a Freesound API token in the environment::

    export FREESOUND_API_TOKEN=...   # https://freesound.org/apiv2/apply/
    uv run --extra cpu python evals/download_freesound.py
    uv run --extra cpu python evals/download_freesound.py --query "sustained vowel ah" --count 5

Searches Freesound for short sung/sustained vowels under a permissive license (Creative Commons 0
by default — no attribution required, so the resulting voices pass the consent gate FR-011 cleanly),
downloads the high-quality preview, resamples to 22,050 Hz mono float32, and writes a donor WAV
under git-ignored ``models/donors/freesound/``. Attribution + license for every fetched sound is
recorded in ``ATTRIBUTION.txt`` alongside the WAVs, and a ready-to-paste ``Voice`` block is printed.

License values accepted by ``--license`` map to Freesound's filter vocabulary:

  - ``cc0``  → "Creative Commons 0" (public domain, no attribution)  [default]
  - ``by``   → "Attribution" (credit required — see ATTRIBUTION.txt)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from voders.constants import SAMPLE_RATE
from voders.voices.models import Voice, VoiceKind

API = "https://freesound.org/apiv2"
FREESOUND_ROOT = Path(__file__).resolve().parents[1] / "models" / "donors" / "freesound"

# Map the friendly --license choice to (Freesound filter value, license string for the Voice entry).
LICENSES = {
    "cc0": ('license:"Creative Commons 0"', "CC0 1.0 (Freesound)"),
    "by": ('license:"Attribution"', "CC BY 4.0 (Freesound) — attribution required"),
}


def _voice_yaml(voice: Voice) -> str:
    """Render a single ``Voice`` as a ready-to-paste YAML list item."""
    return yaml.safe_dump(
        [voice.model_dump(mode="json")],
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug."""
    keep = [c if c.isalnum() else "_" for c in text.lower()]
    return "".join(keep).strip("_")[:40] or "sound"


def _get(url: str, token: str) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted Freesound host)
        return resp.read()


def search(query: str, license_filter: str, count: int, token: str) -> list[dict]:
    """Return up to ``count`` Freesound search hits with preview URLs."""
    params = urllib.parse.urlencode(
        {
            "query": query,
            "filter": f"{license_filter} duration:[0.5 TO 15]",
            "fields": "id,name,username,license,previews,duration",
            "sort": "rating_desc",
            "page_size": count,
        }
    )
    data = json.loads(_get(f"{API}/search/text/?{params}", token))
    return data.get("results", [])[:count]


def fetch_preview(hit: dict, dest: Path, token: str) -> np.ndarray | None:
    """Download a hit's HQ preview, decode to mono float32, resample to SAMPLE_RATE."""
    previews = hit.get("previews") or {}
    url = previews.get("preview-hq-ogg") or previews.get("preview-lq-ogg")
    if not url:
        print(f"  ! {hit['id']} has no downloadable preview, skipping")
        return None
    import io

    raw = _get(url, token)
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = (arr / peak * 0.9).astype(np.float32)
    sf.write(str(dest), arr, SAMPLE_RATE, subtype="FLOAT")
    return arr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch donor vowels from Freesound")
    parser.add_argument("--query", default="sustained sung vowel", help="Freesound search query")
    parser.add_argument(
        "--count", type=int, default=3, help="How many sounds to fetch (default: 3)"
    )
    parser.add_argument(
        "--license",
        choices=sorted(LICENSES),
        default="cc0",
        help="License to restrict to (default: cc0 — no attribution required)",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("FREESOUND_API_TOKEN")
    if not token:
        print(
            "ERROR: set FREESOUND_API_TOKEN (get one at https://freesound.org/apiv2/apply/).",
            file=sys.stderr,
        )
        return 2

    license_filter, license_str = LICENSES[args.license]
    print(f"== searching Freesound for {args.query!r} ({args.license}) ==")
    try:
        hits = search(args.query, license_filter, args.count, token)
    except urllib.error.HTTPError as e:
        print(f"ERROR: Freesound API returned {e.code} {e.reason}", file=sys.stderr)
        return 1
    if not hits:
        print("no matching sounds found — try a different --query or --license")
        return 0

    FREESOUND_ROOT.mkdir(parents=True, exist_ok=True)
    attribution = FREESOUND_ROOT / "ATTRIBUTION.txt"
    voices: list[Voice] = []
    with attribution.open("a", encoding="utf-8") as att:
        for hit in hits:
            voice_id = f"freesound_{hit['id']}_{_slug(hit['name'])}"
            dest = FREESOUND_ROOT / f"{voice_id}.wav"
            if dest.exists():
                print(f"  = {dest.name} exists, skipping")
            else:
                print(
                    f"  + {hit['id']} {hit['name']!r} by {hit['username']} ({hit['duration']:.1f}s)"
                )
                arr = fetch_preview(hit, dest, token)
                if arr is None:
                    continue
                att.write(
                    f'{dest.name}\tFreesound #{hit["id"]} "{hit["name"]}" by {hit["username"]}'
                    f"\t{license_str}\thttps://freesound.org/s/{hit['id']}/\n"
                )
            voices.append(
                Voice(
                    voice_id=voice_id,
                    kind=VoiceKind.DETERMINISTIC_DONOR,
                    license=license_str,
                    consent_verified=True,
                    model_ref=str(dest),
                )
            )

    if voices:
        print("\n" + "=" * 60)
        print("Add these under `voices:` in your run config:")
        print("=" * 60)
        for voice in voices:
            print(_voice_yaml(voice).rstrip())
        print("=" * 60)
        print(f"Attribution recorded in {attribution}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
