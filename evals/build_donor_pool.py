"""Build a single run config whose voice pool is every donor from every data-source method.

This is the systematic entry point that unifies all donor-enrollment methods into one corpus:

  1. synthetic fixtures   evals/fixtures/voices/*.wav     (``evals/make_fixtures.py``)
  2. streamed datasets    models/donors/*.wav             (``download_donors.py``: VocalSet/VCTK)
  3. operator recordings  models/donors/*.wav             (``evals/record_donor.py``)
  4. Freesound fetches    models/donors/freesound/*.wav   (``evals/download_freesound.py``)

It scans those locations, infers each voice's license (Freesound's ``ATTRIBUTION.txt``; known
dataset filenames; a consented default otherwise), and writes a validated :class:`RunConfig` YAML.
Every donor here is consented by construction, so all entries carry ``consent_verified: true`` and
pass the gate (FR-011, SC-008). Run it after any fetch/record step to refresh the pool::

    uv run --extra cpu python evals/build_donor_pool.py     # -> evals/fixtures/donor_pool.yaml

Because it discovers from disk, newly fetched or recorded donors join the pool automatically — no
hand-editing of the config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from voders.config.models import RunConfig
from voders.voices.models import Voice, VoiceKind

REPO = Path(__file__).resolve().parents[1]
SYNTH_DIR = REPO / "evals" / "fixtures" / "voices"
DONORS_DIR = REPO / "models" / "donors"
FREESOUND_DIR = DONORS_DIR / "freesound"
DEFAULT_OUT = REPO / "evals" / "fixtures" / "donor_pool.yaml"

# Licenses for the known streamed-dataset filenames written by evals/download_donors.py.
DATASET_LICENSES = {
    "vocalset_singer": "VocalSet, CC BY 4.0 (Wilkins et al.); streamed sample",
    "vctk_speaker": "VCTK, CC BY 4.0; streamed sample",
}


def _dataset_license(stem: str) -> str:
    """License for a top-level donor wav by filename stem (prefix-aware for multi-singer sets)."""
    if stem in DATASET_LICENSES:
        return DATASET_LICENSES[stem]
    if stem.startswith("vocalset"):  # vocalset_00, vocalset_01, ... (one clip per singer)
        return DATASET_LICENSES["vocalset_singer"]
    if stem.startswith("vctk"):
        return DATASET_LICENSES["vctk_speaker"]
    return "operator-recorded, consented"


def _freesound_licenses() -> dict[str, str]:
    """Map ``<file>.wav`` -> license string from the Freesound ATTRIBUTION.txt (if present)."""
    att = FREESOUND_DIR / "ATTRIBUTION.txt"
    licenses: dict[str, str] = {}
    if att.exists():
        for line in att.read_text(encoding="utf-8").splitlines():
            cols = line.split("\t")
            if len(cols) >= 3:
                licenses[cols[0]] = cols[2]
    return licenses


def _rel(path: Path) -> str:
    """Repo-relative POSIX path for the config (matches the other fixtures)."""
    return path.resolve().relative_to(REPO).as_posix()


def discover_voices() -> list[Voice]:
    """Discover every consented donor voice across all data-source methods (sorted, stable)."""
    voices: list[Voice] = []
    seen: set[str] = set()

    def add(path: Path, license: str) -> None:
        voice_id = path.stem
        if voice_id in seen:
            return
        seen.add(voice_id)
        voices.append(
            Voice(
                voice_id=voice_id,
                kind=VoiceKind.DETERMINISTIC_DONOR,
                license=license,
                consent_verified=True,
                model_ref=_rel(path),
            )
        )

    # 1. Synthetic fixtures.
    for wav in sorted(SYNTH_DIR.glob("*.wav")):
        add(wav, "CC0 synthetic vowel")

    # 2./3. Streamed datasets + operator recordings (top-level models/donors/, non-recursive).
    for wav in sorted(DONORS_DIR.glob("*.wav")):
        add(wav, _dataset_license(wav.stem))

    # 4. Freesound fetches.
    fs_licenses = _freesound_licenses()
    for wav in sorted(FREESOUND_DIR.glob("*.wav")):
        add(wav, fs_licenses.get(wav.name, "CC0 1.0 (Freesound)"))

    return voices


# Label-preserving augmentation profiles (FR-005). Every accepted base render is fanned through
# these, so the corpus carries channel/room/codec variation without touching the labels (SC-006).
AUGMENTATION_PROFILES = [
    # Gentle, label-safe settings: the direct sound stays at t=0 and accompaniment stays well below
    # the vocal, so onsets/f0 survive the validator (keeps SC-007 first-attempt pass high).
    {
        "profile_id": "room_reverb",
        "steps": ["reverb_ir"],
        "params": {"reverb_decay_s": 0.08, "reverb_wet": 0.08},
    },
    {"profile_id": "phone_codec", "steps": ["codec"], "params": {"bitrate_kbps": 24.0}},
    {
        "profile_id": "noisy_codec",
        "steps": ["codec", "accompaniment_mix"],
        "params": {"bitrate_kbps": 32.0, "snr_db": [18.0, 24.0]},
    },
]


def build_config(run_id: str, seed: int, accompaniment: str | None = None) -> RunConfig:
    voices = discover_voices()
    if not voices:
        raise SystemExit(
            "no donor voices found — run `just fixtures`, `just download-donors`, "
            "`just download-freesound`, or `just record-donor` first."
        )
    lanes: dict[str, dict[str, object]] = {
        "deterministic": {"enabled": True, "synth": "world"},
        "augmentation": {"enabled": True},
    }
    if accompaniment:
        # Vocal-conditioned accompaniment (spec 002) as a training augmentation: each accepted donor
        # render (incl. the real VocalSet singer) is layered with instrumental backing that keeps
        # labels valid. Lego keeps the vocal bit-exact and the pad sits below the voice so onsets/f0
        # survive the validator. `fake` is the CPU/CI backend; `acestep` is the real GPU model.
        lanes["accompaniment"] = {
            "enabled": True,
            "mode": "lego",
            "backend": accompaniment,
            "model_id": "ace-step-v1-3.5b",
            "license_policy": ["MIT", "Apache-2.0", "CC-BY-4.0"],
            "target_instrument": "deep upright bass and soft brushed drums, warm low chords",
            "free_time": True,
            "takes": 1,
            "target_snr_db": [9.0, 12.0],
        }
    return RunConfig(
        run_id=run_id,
        master_seed=seed,
        scores="evals/fixtures/scores/*.tsv",
        output_root=f"out/{run_id}",
        voices=voices,
        lanes=lanes,
        augmentation_profiles=AUGMENTATION_PROFILES,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the unified donor-pool run config")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output config path")
    parser.add_argument("--run-id", default="donor_pool", help="run_id for the config")
    parser.add_argument("--seed", type=int, default=20260627, help="master_seed")
    parser.add_argument(
        "--accompaniment",
        choices=["fake", "acestep"],
        default=None,
        help="also enable the vocal-conditioned accompaniment lane (spec 002) with this backend",
    )
    args = parser.parse_args(argv)

    config = build_config(args.run_id, args.seed, accompaniment=args.accompaniment)
    header = (
        "# Unified donor pool — every consented donor across all data-source methods (synthetic\n"
        "# fixtures, VocalSet/VCTK, operator recordings, Freesound). GENERATED by\n"
        "# evals/build_donor_pool.py; re-run it after any fetch/record to refresh.\n"
    )
    body = yaml.safe_dump(
        config.model_dump(mode="json"),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    Path(args.out).write_text(header + body, encoding="utf-8")

    def count(pred) -> int:  # noqa: ANN001
        return sum(1 for v in config.voices if pred(v.license))

    by_source = {
        "synthetic": count(lambda lic: lic == "CC0 synthetic vowel"),
        "freesound": count(lambda lic: "Freesound" in lic),
        "dataset": count(lambda lic: "CC BY 4.0" in lic and "Freesound" not in lic),
        "recorded": count(lambda lic: lic == "operator-recorded, consented"),
    }
    print(f"wrote {args.out} — {len(config.voices)} donor voices")
    for src, n in by_source.items():
        if n:
            print(f"  {src:>10}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
