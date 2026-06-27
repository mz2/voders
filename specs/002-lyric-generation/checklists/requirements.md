# Specification Quality Checklist: Optional Lyric Generation & Phonetic Diversity

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-06-27
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`.
- Resolved during authoring (informed defaults, documented in Assumptions rather than left as
  clarifications):
  - **Lyric storage shape** — adopted the `lyrics`-branch prototype decision (optional per-note
    `lyric` + optional 4th score column) instead of a sidecar file; backward-compatibility is an
    explicit requirement (FR-001/FR-002, SC-001).
  - **Scope of the generated (model-based) source** — kept in scope but isolated as the
    lowest-priority, fully droppable story (US4 / P4), so the feature's core value lands via the
    supplied + automatic sources alone.
  - **Theming channel** — restricted to an operator-supplied theme string; never read from
    annotations (FR-011), per the operator constraint.
- Tool/model names (espeak-ng, DiffSinger, specific lyric models, CMUdict) are deliberately deferred
  to `/speckit-plan`; the spec stays at outcome/altitude level.

### Amendment 2026-06-27 — "no timing or pitch shift from added vocal synthesis"

- Added as a **differential (lyric-on vs lyric-off) null-difference guarantee** rather than a new
  feature: Clarifications session entry; US1 acceptance scenario 4; two edge cases (net lead-in/latency,
  articulation perturbing pitch); FR-015 (pitch neutrality), FR-016 (timing neutrality), FR-017
  (constant-delay compensation), FR-018 (differential harness); SC-008 (no pitch shift), SC-009 (no
  timing shift); two assumptions. Dependent docs synced: plan eval strategy + constraints, research
  Decision L4 addendum, quickstart eval table, data-model diagnostics note.
- Re-validated: all checklist items still pass; 0 `[NEEDS CLARIFICATION]`. The new criteria are
  measurable (±25 cents / 10 ms, ≥99%) and technology-agnostic.
- Scope note: amended the in-progress 002 spec in place (no new `003-…` feature/branch), because the
  constraint is intrinsic to "the vocal synthesis being added" — i.e. this feature.

### Remediation 2026-06-27 — post-`/speckit-analyze` fixes

Applied the analyze report's actionable findings (no new feature; edits only):
- **I1/C1 (fixture-name drift)**: tasks now use the quickstart's canonical fixture names —
  `lyrics-supplied.yaml` (T013, US1), `lyrics-smoke.yaml`=automatic (T026, US2),
  `lyrics-generated.yaml` (T037, US4), `lyrics-offbaseline.yaml` (baseline). plan eval-strategy fixture
  list updated to match.
- **C2 (FR-005 untested)**: T002 now adds a failing `tests/unit/test_lyrics_imports.py` asserting
  `import voders.lyrics` pulls in neither `torch` nor `phonemizer` — FR-005 is now a gated test, not
  just a lint guard.
- **U1 (melisma)**: v1 accepts only `melisma: per_note`; `sustain_ties` is reserved and rejected by
  config validation (T010 contract test + T011 impl; spec edge case + Assumptions + data-model row
  updated) so no unimplemented mode is silently accepted.
- **V1 (SC-005 not eval-gated)**: T012 eval suite now asserts SC-005 (zero dropped/added/shifted note
  labels across count mismatches).
- **T2 (plan tree)**: plan source tree now lists `src/voders/lyrics/data/en_cv.txt`.
- Re-validated: FR/SC coverage remains 100%; FR-005 upgraded from test-light to tested; no new
  `[NEEDS CLARIFICATION]`.
