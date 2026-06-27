"""Evaluation harness — Success-Criteria pass/fail over a produced manifest (Constitution IV).

Single-command verdict wrapped by ``voders eval``. Computes the gated Success Criteria and returns
a structured result; the CLI prints the table and sets the exit code.

Gated here: SC-001 (onset), SC-002 (offset), SC-007 (first-attempt pass), SC-008 (consent).
Extended by later user stories: SC-004 (timbre identities), SC-006 (augmentation coverage),
SC-009 (isolated reproducibility), SC-010 (re-derived onset deviation).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from voders.manifest.io import load_manifest
from voders.manifest.models import ProvenanceRecord, VerdictStatus


@dataclass
class CriterionResult:
    sc: str
    description: str
    value: float
    threshold: float
    passed: bool
    gated: bool = True
    detail: str = ""


@dataclass
class EvalReport:
    criteria: list[CriterionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.criteria if c.gated)

    def add(self, c: CriterionResult) -> None:
        self.criteria.append(c)


def _accepted(records: Iterable[ProvenanceRecord]) -> list[ProvenanceRecord]:
    return [r for r in records if r.verdict.status == VerdictStatus.ACCEPTED]


def _synthesized(records: Iterable[ProvenanceRecord]) -> list[ProvenanceRecord]:
    # "Synthesized" excludes consent refusals (those never reached a renderer).
    return [r for r in records if r.verdict.status != VerdictStatus.LICENSE_REFUSED]


def _fraction(num: int, den: int) -> float:
    return 1.0 if den == 0 else num / den


def sc001_onset(records: list[ProvenanceRecord]) -> CriterionResult:
    acc = _accepted(records)
    ok = sum(1 for r in acc if r.verdict.onset_ok)
    value = _fraction(ok, len(acc))
    return CriterionResult(
        "SC-001",
        "onset within 50 ms ≥99% of accepted",
        value,
        0.99,
        value >= 0.99,
        detail=f"{ok}/{len(acc)} accepted",
    )


def sc002_offset(records: list[ProvenanceRecord]) -> CriterionResult:
    acc = _accepted(records)
    ok = sum(1 for r in acc if r.verdict.offset_ok)
    value = _fraction(ok, len(acc))
    return CriterionResult(
        "SC-002",
        "offset within max(50 ms,20%) ≥99% of accepted",
        value,
        0.99,
        value >= 0.99,
        detail=f"{ok}/{len(acc)} accepted",
    )


def sc007_first_pass(records: list[ProvenanceRecord]) -> CriterionResult:
    syn = _synthesized(records)
    acc = _accepted(syn)
    value = _fraction(len(acc), len(syn))
    return CriterionResult(
        "SC-007",
        "first-attempt validator pass ≥95%",
        value,
        0.95,
        value >= 0.95,
        detail=f"{len(acc)}/{len(syn)} synthesized",
    )


def sc008_consent(records: list[ProvenanceRecord]) -> CriterionResult:
    bad = [r for r in _accepted(records) if not r.consent_verified]
    value = float(len(bad))
    return CriterionResult(
        "SC-008",
        "zero unconsented voices in accepted corpus",
        value,
        0.0,
        len(bad) == 0,
        detail=f"{len(bad)} unconsented accepted",
    )


def sc004_timbre_identities(records: list[ProvenanceRecord]) -> CriterionResult:
    """Distinct timbre identities = (voice, augmentation profile) among accepted (SC-004)."""
    identities = {(r.voice_id, r.augmentation_profile or "") for r in _accepted(records)}
    value = float(len(identities))
    # Scale-dependent: gated only once the corpus is large enough to plausibly hit the target.
    gated = len(_accepted(records)) >= 1000
    return CriterionResult(
        "SC-004",
        "≥1,000 distinct timbre identities",
        value,
        1000.0,
        value >= 1000.0,
        gated=gated,
        detail=f"{len(identities)} identities",
    )


def sc006_augmentation_coverage(records: list[ProvenanceRecord]) -> CriterionResult:
    """Fraction of accepted samples carrying ≥1 production-style augmentation (SC-006)."""
    acc = _accepted(records)
    augmented = sum(1 for r in acc if r.augmentation_profile)
    value = _fraction(augmented, len(acc))
    # Gated only when the augmentation lane was part of the run.
    gated = any(r.lane == "augmentation" for r in records)
    return CriterionResult(
        "SC-006",
        "≥60% of accepted carry an augmentation",
        value,
        0.60,
        value >= 0.60,
        gated=gated,
        detail=f"{augmented}/{len(acc)} accepted",
    )


def sc010_rederived_onset(records: list[ProvenanceRecord]) -> CriterionResult:
    """Re-derived onset deviation <50 ms in ≥90% of accepted expressive-lane notes (SC-010)."""
    svs = [r for r in _accepted(records) if r.lane == "svs"]
    within = sum(1 for r in svs if r.verdict.max_onset_dev_ms < 50.0)
    value = _fraction(within, len(svs))
    gated = len(svs) > 0
    return CriterionResult(
        "SC-010",
        "re-derived onset deviation <50 ms in ≥90% (svs lane)",
        value,
        0.90,
        value >= 0.90,
        gated=gated,
        detail=f"{within}/{len(svs)} svs accepted",
    )


def sc009_reproducibility(manifest_path: str, max_samples: int = 3) -> CriterionResult:
    """Re-render a sampled subset in isolation and assert the SC-009 tolerance per lane.

    Loads ``config.resolved.yaml`` from the manifest directory; if absent, the criterion is
    reported as informational (cannot reproduce without the resolved config).
    """
    from pathlib import Path

    manifest = Path(manifest_path)
    config_path = manifest.parent / "config.resolved.yaml"
    # Exclude post-acceptance fan-out stages (augmentation + the neural accompaniment lane): they
    # are reproduced under the same-verdict neural tier, not by a bit-exact renderer re-render.
    _post_stages = {"augmentation", "accompaniment"}
    records = [r for r in _accepted(load_manifest(manifest_path)) if r.lane not in _post_stages]
    if not config_path.exists() or not records:
        return CriterionResult(
            "SC-009",
            "isolated re-render matches within tolerance",
            1.0,
            1.0,
            True,
            gated=False,
            detail="skipped (no resolved config or no samples)",
        )

    from voders.config.loader import load_config
    from voders.corpus.reproduce import reproduce_sample
    from voders.render.registry import build_lanes
    from voders.validate.validator import Validator

    config = load_config(config_path)
    lanes = build_lanes(config)
    validator = Validator(config.validator)
    sample = records[:: max(1, len(records) // max_samples)][:max_samples]
    matched = 0
    for rec in sample:
        res = reproduce_sample(manifest.parent, rec, config, lanes, validator)
        matched += int(res.matched)
    value = _fraction(matched, len(sample))
    return CriterionResult(
        "SC-009",
        "isolated re-render matches within tolerance",
        value,
        1.0,
        matched == len(sample),
        detail=f"{matched}/{len(sample)} reproduced",
    )


# --- Accompaniment stage (spec 002) Success Criteria ---------------------------------------------
# Distinct "ACC-" ids so the 001 SC-001..SC-010 above are untouched; descriptions cite the 002 SC.

_NONCOMMERCIAL_LICENSE_MARKERS = ("-NC", "NONCOMMERCIAL", "NON-COMMERCIAL", "CC-BY-NC")


def _accompaniment(records: Iterable[ProvenanceRecord]) -> list[ProvenanceRecord]:
    return [r for r in records if r.lane == "accompaniment" and r.accompaniment is not None]


def acc_sc001_bit_exact(records: list[ProvenanceRecord]) -> CriterionResult:
    """002 SC-001: every accepted Lego sample keeps the vocal bit-exact (vocal-preserving)."""
    lego = [r for r in _accepted(_accompaniment(records)) if r.accompaniment.mode == "lego"]
    bad = [r for r in lego if not r.accompaniment.vocal_bit_exact]
    return CriterionResult(
        "ACC-001",
        "002 SC-001: Lego accepted keep vocal bit-exact",
        float(len(bad)),
        0.0,
        len(bad) == 0,
        gated=len(lego) > 0,
        detail=f"{len(bad)}/{len(lego)} lego not bit-exact",
    )


def acc_sc002_timing(records: list[ProvenanceRecord]) -> CriterionResult:
    """002 SC-002: ≥99% of accepted Complete samples keep onset+offset within tolerance on mix."""
    comp = [r for r in _accepted(_accompaniment(records)) if r.accompaniment.mode == "complete"]
    ok = sum(1 for r in comp if r.verdict.onset_ok and r.verdict.offset_ok)
    value = _fraction(ok, len(comp))
    return CriterionResult(
        "ACC-002",
        "002 SC-002: Complete onset+offset within tol ≥99%",
        value,
        0.99,
        value >= 0.99,
        gated=len(comp) > 0,
        detail=f"{ok}/{len(comp)} complete accepted",
    )


def acc_sc004_provenance(records: list[ProvenanceRecord]) -> CriterionResult:
    """002 SC-004: complete provenance, no non-commercial license, CC-BY carries attribution."""
    acc = _accepted(_accompaniment(records))
    violations = 0
    for r in acc:
        a = r.accompaniment
        complete = bool(a.model_id and a.model_license and a.mode and a.source_vocal_sample_id)
        lic = a.model_license.upper()
        noncommercial = any(m in lic for m in _NONCOMMERCIAL_LICENSE_MARKERS)
        cc_by_needs_attr = lic.startswith("CC-BY") and not noncommercial and not a.attribution_text
        if not complete or noncommercial or cc_by_needs_attr:
            violations += 1
    return CriterionResult(
        "ACC-004",
        "002 SC-004: provenance complete, license in policy, CC-BY attributed",
        float(violations),
        0.0,
        violations == 0,
        gated=len(acc) > 0,
        detail=f"{violations}/{len(acc)} accepted with a provenance/license violation",
    )


def acc_sc005_note_shift(records: list[ProvenanceRecord]) -> CriterionResult:
    """002 SC-005: no accepted sample's note shifted beyond the validator tolerance."""
    from voders.constants import ONSET_TOLERANCE_MS

    acc = _accepted(_accompaniment(records))
    bad = [r for r in acc if r.accompaniment.max_note_shift_ms > ONSET_TOLERANCE_MS]
    return CriterionResult(
        "ACC-005",
        "002 SC-005: zero note-timing shift beyond tolerance",
        float(len(bad)),
        0.0,
        len(bad) == 0,
        gated=len(acc) > 0,
        detail=f"{len(bad)}/{len(acc)} accepted shifted > {ONSET_TOLERANCE_MS:.0f} ms",
    )


def acc_sc008_stems(records: list[ProvenanceRecord]) -> CriterionResult:
    """002 SC-008: every accepted Lego sample retains a re-mixable stem."""
    lego = [r for r in _accepted(_accompaniment(records)) if r.accompaniment.mode == "lego"]
    bad = [r for r in lego if not r.accompaniment.stem_available]
    return CriterionResult(
        "ACC-008",
        "002 SC-008: Lego accepted retain a re-mixable stem",
        float(len(bad)),
        0.0,
        len(bad) == 0,
        gated=len(lego) > 0,
        detail=f"{len(bad)}/{len(lego)} lego without a stem",
    )


def evaluate(manifest_path: str, *, reproduce: bool = True) -> EvalReport:
    records = load_manifest(manifest_path)
    report = EvalReport()
    report.add(sc001_onset(records))
    report.add(sc002_offset(records))
    report.add(sc007_first_pass(records))
    report.add(sc008_consent(records))
    report.add(sc004_timbre_identities(records))
    report.add(sc006_augmentation_coverage(records))
    report.add(sc010_rederived_onset(records))
    # Accompaniment stage (spec 002) — gated only when accompaniment samples are present.
    report.add(acc_sc001_bit_exact(records))
    report.add(acc_sc002_timing(records))
    report.add(acc_sc004_provenance(records))
    report.add(acc_sc005_note_shift(records))
    report.add(acc_sc008_stems(records))
    if reproduce:
        report.add(sc009_reproducibility(manifest_path))
    return report


def format_report(report: EvalReport) -> str:
    """Render the SC pass/fail table."""
    lines = [
        f"{'SC':<8} {'result':<6} {'value':>8} {'thr':>8}  description",
        "-" * 72,
    ]
    for c in report.criteria:
        mark = "PASS" if c.passed else "FAIL"
        gate = "" if c.gated else " (info)"
        lines.append(
            f"{c.sc:<8} {mark:<6} {c.value:>8.3f} {c.threshold:>8.3f}  {c.description}{gate}"
        )
        if c.detail:
            lines.append(f"{'':<24}{c.detail}")
    lines.append("-" * 72)
    lines.append("OVERALL: " + ("PASS" if report.passed else "FAIL"))
    return "\n".join(lines)
