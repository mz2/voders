"""``voders audit`` — license/consent audit over the manifest (FR-011, SC-008)."""

from __future__ import annotations

from voders.manifest.io import load_manifest
from voders.manifest.models import VerdictStatus


def audit_command(manifest_path: str) -> int:
    records = load_manifest(manifest_path)
    voices: dict[str, str] = {}
    refused = 0
    unconsented_accepted = 0
    for r in records:
        voices[r.voice_id] = r.voice_license
        if r.verdict.status == VerdictStatus.LICENSE_REFUSED:
            refused += 1
        if r.verdict.status == VerdictStatus.ACCEPTED and not r.consent_verified:
            unconsented_accepted += 1

    print(f"donor voices present: {len(voices)}")
    for vid, lic in sorted(voices.items()):
        print(f"  {vid}: {lic}")
    print(f"consent refusals logged: {refused}")
    print(f"unconsented voices in accepted corpus: {unconsented_accepted}")
    if unconsented_accepted > 0:
        print("AUDIT FAILED: unconsented voice reached the accepted corpus (SC-008)")
        return 1
    print("AUDIT PASSED (SC-008)")
    return 0
