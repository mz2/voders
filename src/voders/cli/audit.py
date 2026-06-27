"""``voders audit`` — license/consent audit over the manifest (FR-011, SC-008; FR-006a)."""

from __future__ import annotations

from voders.manifest.io import load_manifest
from voders.manifest.models import VerdictStatus

# Markers of a non-commercial license, which the policy excludes (spec 002 FR-006).
_NONCOMMERCIAL_MARKERS = ("-NC", "NONCOMMERCIAL", "NON-COMMERCIAL", "CC-BY-NC")


def _audit_accompaniment(records: list) -> tuple[int, list[str]]:  # noqa: ANN001
    """Check accepted accompaniment samples' license + attribution (FR-006/006a).

    Returns ``(violation_count, lines)``. A violation is a non-commercial/empty license or a
    CC-BY-class model whose required attribution text is missing.
    """
    accepted = [
        r
        for r in records
        if r.lane == "accompaniment"
        and r.accompaniment is not None
        and r.verdict.status == VerdictStatus.ACCEPTED
    ]
    lines: list[str] = []
    violations = 0
    if not accepted:
        return 0, lines
    models: dict[str, str] = {}
    for r in accepted:
        a = r.accompaniment
        models[a.model_id] = a.model_license
        lic = a.model_license.upper()
        noncommercial = (not a.model_license) or any(m in lic for m in _NONCOMMERCIAL_MARKERS)
        cc_by_missing_attr = (
            lic.startswith("CC-BY") and not noncommercial and not a.attribution_text
        )
        if noncommercial:
            violations += 1
            lines.append(
                f"  VIOLATION {r.sample_id}: non-commercial/empty license {a.model_license!r}"
            )
        elif cc_by_missing_attr:
            violations += 1
            lines.append(f"  VIOLATION {r.sample_id}: CC-BY model missing attribution text")
    lines.insert(0, f"accompaniment models present: {len(models)}")
    for mid, lic in sorted(models.items()):
        lines.insert(1, f"  {mid}: {lic}")
    return violations, lines


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

    acc_violations, acc_lines = _audit_accompaniment(records)
    for line in acc_lines:
        print(line)

    failed = unconsented_accepted > 0 or acc_violations > 0
    if unconsented_accepted > 0:
        print("AUDIT FAILED: unconsented voice reached the accepted corpus (SC-008)")
    if acc_violations > 0:
        print(
            f"AUDIT FAILED: {acc_violations} accompaniment "
            "license/attribution violation(s) (FR-006a)"
        )
    if failed:
        return 1
    print("AUDIT PASSED (SC-008, FR-006a)")
    return 0
