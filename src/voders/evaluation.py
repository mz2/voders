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
from pathlib import Path

from voders.manifest.io import load_manifest
from voders.manifest.models import ProvenanceRecord, VerdictStatus
from voders.scores.models import Note


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


# --------------------------------------------------------------------------------------------------
# Score-augmentation suite (feature 003): SC-001..SC-009 over a produced manifest.
#
# Differential criteria read each variant record's ``base_score_id`` to locate its base in the same
# manifest and compare label rows. Determinism is checked at the label level / via in-process
# re-expansion — the WORLD renderer is bit-exact in-process but not across process invocations.
# --------------------------------------------------------------------------------------------------


def _is_variant(r: ProvenanceRecord) -> bool:
    return r.base_score_id is not None


def _is_original(r: ProvenanceRecord) -> bool:
    return r.base_score_id is None and r.lane != "score_augmentation"


def _notes_for(root: Path, record: ProvenanceRecord | None) -> list[Note] | None:
    from pathlib import Path

    from voders.scores.parse import parse_tsv

    if record is None or not record.score_path:
        return None
    p = Path(root) / record.score_path
    if not p.exists():
        return None
    try:
        return parse_tsv(p).score.notes
    except Exception:
        return None


def _vacuous(sc: str, description: str, detail: str) -> CriterionResult:
    return CriterionResult(sc, description, 1.0, 1.0, True, gated=False, detail=detail)


def sc001_off_safe(records: list[ProvenanceRecord], root: Path) -> CriterionResult:
    """Opt-in safety: originals are untouched — inert score-aug fields, never under augmented/."""
    originals = [r for r in records if _is_original(r)]
    bad = [
        r
        for r in originals
        if r.score_aug_axis is not None
        or r.score_aug_profile is not None
        or r.base_score_id is not None
        or (r.score_path and "augmented/" in r.score_path)
    ]
    ok = len(originals) - len(bad)
    value = _fraction(ok, len(originals))
    return CriterionResult(
        "SC-001",
        "feature-off-safe: originals carry inert score-aug fields, never under augmented/",
        value,
        1.0,
        not bad,
        detail=f"{ok}/{len(originals)} originals clean",
    )


def sc002_transpose_exact(records: list[ProvenanceRecord], root: Path) -> CriterionResult:
    """Every transpose variant: pitch shifted by exactly the offset; timing identical; pitch in range."""  # noqa: E501
    bases = {(r.score_id, r.lane, r.voice_id): r for r in records if _is_original(r)}
    variants = [r for r in _accepted(records) if r.score_aug_axis == "transpose" and _is_variant(r)]
    if not variants:
        return _vacuous("SC-002", "transpose exactness", "no transpose variants")
    ok = 0
    for v in variants:
        offset = int((v.score_aug_transform or "t+0")[1:])
        base = bases.get((v.base_score_id or "", v.lane, v.voice_id))
        vn = _notes_for(root, v)
        bn = _notes_for(root, base)
        if not vn or not bn or len(vn) != len(bn):
            continue
        good = all(
            a.pitch_midi == b.pitch_midi + offset
            and a.onset_s == b.onset_s
            and a.offset_s == b.offset_s
            and 0 <= a.pitch_midi <= 127
            for a, b in zip(vn, bn, strict=True)
        )
        ok += int(good)
    value = _fraction(ok, len(variants))
    return CriterionResult(
        "SC-002",
        "transpose: pitch shifted by exactly the offset, timing identical, pitch in 0..127",
        value,
        1.0,
        ok == len(variants),
        detail=f"{ok}/{len(variants)} transpose variants exact",
    )


def sc003_humanize_valid(
    records: list[ProvenanceRecord], root: Path, max_dev_by_profile: dict[str, float]
) -> CriterionResult:
    """Every humanise variant is a valid monophonic score with deviations within the budget."""
    from voders.scores.models import Score

    bases = {(r.score_id, r.lane, r.voice_id): r for r in records if _is_original(r)}
    variants = [r for r in records if r.score_aug_axis == "humanize" and _is_variant(r)]
    if not variants:
        return _vacuous("SC-003", "humanise validity + budget", "no humanise variants")
    ok = 0
    for v in variants:
        vn = _notes_for(root, v)
        base = bases.get((v.base_score_id or "", v.lane, v.voice_id))
        bn = _notes_for(root, base)
        if not vn:
            continue
        valid = Score(score_id="x", notes=list(vn)).is_monophonic() and all(
            n.duration_s > 0 for n in vn
        )
        budget = max_dev_by_profile.get(v.score_aug_profile or "")
        within = True
        if bn and budget is not None and len(vn) == len(bn):
            within = all(
                abs(a.onset_s - b.onset_s) <= budget + 1e-9
                and abs(a.offset_s - b.offset_s) <= budget + 1e-9
                for a, b in zip(vn, bn, strict=True)
            )
        ok += int(valid and within)
    value = _fraction(ok, len(variants))
    return CriterionResult(
        "SC-003",
        "humanise: valid monophonic score, deviations within max-deviation budget",
        value,
        1.0,
        ok == len(variants),
        detail=f"{ok}/{len(variants)} humanise variants valid",
    )


def sc004_volume_widens(records: list[ProvenanceRecord], root: Path) -> CriterionResult:
    """Volume variants widen per-note levels; (onset,offset,pitch) labels stay identical."""
    from pathlib import Path

    import numpy as np

    from voders.audio import read_wav
    from voders.constants import SAMPLE_RATE

    bases = {(r.score_id, r.lane, r.voice_id): r for r in _accepted(records) if _is_original(r)}
    variants = [r for r in _accepted(records) if r.score_aug_axis == "volume" and _is_variant(r)]
    if not variants:
        return _vacuous("SC-004", "volume widens level distribution", "no volume variants")

    def per_note_rms(rec: ProvenanceRecord) -> list[float]:
        notes = _notes_for(root, rec)
        wpath = Path(root) / rec.audio_path
        if not notes or not wpath.exists():
            return []
        audio, _ = read_wav(wpath)
        out = []
        for n in notes:
            a = int(n.onset_s * SAMPLE_RATE)
            b = min(int(n.offset_s * SAMPLE_RATE), audio.size)
            seg = audio[a:b]
            out.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0)
        return out

    widened = 0
    labels_ok = 0
    for v in variants:
        base = bases.get((v.base_score_id or "", v.lane, v.voice_id))
        if base is None:
            continue
        vn = _notes_for(root, v)
        bn = _notes_for(root, base)
        if vn and bn and len(vn) == len(bn):
            labels_ok += int(
                all(
                    a.onset_s == b.onset_s
                    and a.offset_s == b.offset_s
                    and a.pitch_midi == b.pitch_midi
                    for a, b in zip(vn, bn, strict=True)
                )
            )
        v_rms = per_note_rms(v)
        b_rms = per_note_rms(base)
        if len(v_rms) > 1 and len(b_rms) > 1:
            widened += int(float(np.std(v_rms)) > float(np.std(b_rms)))
    value = _fraction(widened, len(variants))
    passed = widened == len(variants) and labels_ok == len(variants)
    return CriterionResult(
        "SC-004",
        "volume: wider per-note level spread than base, labels byte-identical",
        value,
        1.0,
        passed,
        detail=f"{widened}/{len(variants)} widened, {labels_ok}/{len(variants)} labels identical",
    )


def sc005_reexpand_identical(manifest_path, records, root: Path) -> CriterionResult:
    """Re-expanding each base in-process reproduces every variant's label .tsv byte-for-byte."""
    import glob
    from pathlib import Path

    config_path = Path(root) / "config.resolved.yaml"
    variants = [r for r in records if _is_variant(r) and r.score_path]
    if not config_path.exists() or not variants:
        return _vacuous("SC-005", "re-expansion byte-identical", "skipped (no config or variants)")

    from voders.config.loader import load_config
    from voders.scoreaug.expand import expand_full
    from voders.scores.parse import parse_tsv, serialize_score
    from voders.seeds import derive_seed

    config = load_config(config_path)
    profiles = {p.profile_id: p for p in config.score_augmentation}
    paths = sorted(glob.glob(config.scores)) or sorted(
        glob.glob(str(Path(config.scores) / "*.tsv"))
    )
    bases = {Path(p).stem: parse_tsv(p) for p in paths}

    ok = 0
    checked = 0
    for v in variants:
        prof = profiles.get(v.score_aug_profile)
        base = bases.get(v.base_score_id or "")
        if prof is None or base is None:
            continue
        seed = derive_seed(config.master_seed, base.score.score_id, "score_aug", prof.profile_id)
        produced = {var.transform: var for var in expand_full(base, prof, seed).variants}
        var = produced.get(v.score_aug_transform or "")
        disk = Path(root) / v.score_path
        if var is None or not disk.exists():
            continue
        checked += 1
        ok += int(serialize_score(var.score) == disk.read_bytes())
    if not checked:
        return _vacuous("SC-005", "re-expansion byte-identical", "no comparable variants")
    value = _fraction(ok, checked)
    return CriterionResult(
        "SC-005",
        "re-expansion reproduces every variant label byte-for-byte",
        value,
        1.0,
        ok == checked,
        detail=f"{ok}/{checked} variants reproduced",
    )


def sc006_lineage(records: list[ProvenanceRecord]) -> CriterionResult:
    """Every variant records base id + profile + seed + transform, and is path-classifiable."""
    variants = [r for r in records if _is_variant(r)]
    if not variants:
        return _vacuous("SC-006", "variant lineage present", "no variants")
    ok = 0
    for v in variants:
        complete = (
            v.base_score_id
            and v.score_aug_profile
            and v.score_aug_transform
            and v.score_aug_seed is not None
            and v.score_aug_axis
        )
        classifiable = (not v.score_path) or ("augmented/" in v.score_path)
        ok += int(bool(complete) and classifiable)
    value = _fraction(ok, len(variants))
    return CriterionResult(
        "SC-006",
        "variant lineage: base id + profile + seed + transform present, path-classifiable",
        value,
        1.0,
        ok == len(variants),
        detail=f"{ok}/{len(variants)} variants fully traceable",
    )


def sc007_coverage(manifest_path: str, records: list[ProvenanceRecord]) -> CriterionResult:
    """Stats report a score-augmentation coverage axis; effective count = accepted/originals."""
    from voders.corpus.stats import build_stats

    block = build_stats(manifest_path).get("score_augmentation", {})
    acc = _accepted(records)
    variants = [r for r in acc if _is_variant(r)]
    originals = [r for r in acc if not _is_variant(r)]
    if not variants:
        return _vacuous("SC-007", "coverage axis reported", "no variants")
    expected_mult = _fraction(len(acc), len(originals)) if originals else 0.0
    ok = (
        bool(block.get("enabled"))
        and bool(block.get("by_transform"))
        and abs(float(block.get("effective_multiplier", 0.0)) - expected_mult) < 1e-9
    )
    return CriterionResult(
        "SC-007",
        "coverage axis reported; effective multiplier = accepted/originals",
        1.0 if ok else 0.0,
        1.0,
        ok,
        detail=f"mult={block.get('effective_multiplier')}, transforms={block.get('by_transform')}",
    )


def sc009_separation(records: list[ProvenanceRecord]) -> CriterionResult:
    """Originals and variants never collide on disk; every file path is unique and foldered."""
    paths = [r.score_path for r in records if r.score_path] + [
        r.audio_path for r in records if r.audio_path
    ]
    unique = len(paths) == len(set(paths))
    misfiled = []
    for r in records:
        if not r.score_path:
            continue
        in_aug = "augmented/" in r.score_path
        if _is_variant(r) and not in_aug:
            misfiled.append(r.sample_id)
        if _is_original(r) and in_aug:
            misfiled.append(r.sample_id)
    ok = unique and not misfiled
    return CriterionResult(
        "SC-009",
        "originals vs variants physically separated, zero path collisions",
        1.0 if ok else 0.0,
        1.0,
        ok,
        detail=f"unique_paths={unique}, misfiled={len(misfiled)}",
    )


def evaluate_score_aug(manifest_path: str) -> EvalReport:
    """The feature-003 suite: SC-001..SC-009 over a produced manifest."""
    from pathlib import Path

    records = load_manifest(manifest_path)
    root = Path(manifest_path).parent

    max_dev_by_profile: dict[str, float] = {}
    config_path = root / "config.resolved.yaml"
    if config_path.exists():
        from voders.config.loader import load_config

        for p in load_config(config_path).score_augmentation:
            if p.humanize_time is not None:
                max_dev_by_profile[p.profile_id] = p.humanize_time.max_dev_s

    report = EvalReport()
    report.add(sc001_off_safe(records, root))
    report.add(sc002_transpose_exact(records, root))
    report.add(sc003_humanize_valid(records, root, max_dev_by_profile))
    report.add(sc004_volume_widens(records, root))
    report.add(sc005_reexpand_identical(manifest_path, records, root))
    report.add(sc006_lineage(records))
    report.add(sc007_coverage(manifest_path, records))
    report.add(sc009_separation(records))
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
