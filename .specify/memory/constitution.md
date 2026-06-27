<!--
Sync Impact Report
==================
Version change: 1.1.0 → 1.2.0
Bump rationale: MINOR — a new core principle (V. Clear, Audience-Aware Writing) is added. No existing principle is removed or reversed.

Note on git history: v1.1.0 was never committed (working-tree-only). The next commit will encompass v1.2.0 relative to the committed v1.0.0 baseline, i.e. it adds Principles IV and V together. The intermediate v1.1.0 number is preserved in this report so the principle-addition sequence stays traceable.

Modified principles:
- (none renamed or removed)

Added sections:
- Core Principles: V. Clear, Audience-Aware Writing

Removed sections:
- (none)

Templates requiring updates:
- ✅ `.specify/memory/constitution.md` — written
- ✅ `.specify/templates/plan-template.md` — no edit; the writing principle applies to plan prose at authoring time and is enforced by reviewers, not by template structure
- ✅ `.specify/templates/spec-template.md` — no edit; same reasoning
- ✅ `.specify/templates/tasks-template.md` — no edit; same reasoning
- ✅ `CLAUDE.md` — no edit required

Follow-up TODOs:
- The earlier prose of this constitution (Principles I–IV) and the existing `specs/001-synthetic-singing-corpus/spec.md` predate Principle V. They are not retroactively rewritten in this amendment. A future PATCH amendment MAY tighten them; until then, the principle is forward-looking.
-->

# Voders Constitution

## Core Principles

### I. Red-Green-Refactor TDD (NON-NEGOTIABLE)

Every behavioural change MUST start as a failing test. The canonical loop is **Red → Green → Refactor**: write the smallest test that captures the desired behaviour and confirm it fails for the right reason, write the smallest code that makes it pass, then refactor without changing observable behaviour. No production code may be added or modified without a test that was first observed to fail. Bug fixes MUST begin with a regression test that reproduces the bug. The "tests are optional" wording in `.specify/templates/tasks-template.md` is overridden by this principle: tests-first is the default in every feature that touches behaviour.

**Rationale:** TDD is the cheapest way to keep design pressure on the code, to prevent regressions, and to make refactoring safe. Allowing exceptions ("just this once") is how teams accumulate untestable code; the discipline only works if it is non-negotiable.

### II. Zero-Warning Linting (NON-NEGOTIABLE)

The project's configured linters and formatters MUST run clean — zero errors and zero warnings — on every commit and on every CI run. Suppressing or downgrading a lint rule requires an inline justification comment naming the rule, the reason, and either a removal date or a follow-up issue. Adding a new lint suppression in a PR that does not otherwise touch the suppressed code is forbidden. Mass auto-disable, blanket ignore files, and "warnings are okay" are not allowed.

**Rationale:** Warnings that linger train the team to ignore the lint output, which is exactly when real problems slip through. A clean lint baseline preserves the signal-to-noise ratio of every future lint run.

### III. Documentation Stays Current with the Repo

Documentation that lives in this repository — `README.md`, `CLAUDE.md`, `docs/**`, specs under `specs/**`, the constitution itself, and per-feature plans/tasks — MUST be updated in the same change set as the code or behaviour they describe. If a public command, configuration key, file format, or invariant changes, the corresponding documentation lines change in the same PR. Stale or contradictory documentation MUST be deleted, not left "for later". Reviewers MUST reject PRs whose code-versus-docs delta is incoherent.

**Rationale:** Documentation drift is silent and compounds. The only sustainable way to keep docs honest is to make them part of the change, not a follow-up.

### IV. Evaluation-First Iteration (NON-NEGOTIABLE)

Before implementing any non-trivial feature, the engineer (human or agent) MUST first decide and check in *both* of the following:

- **How to run it.** A reproducible command, script, harness, fixture, or notebook that exercises the feature end-to-end on representative inputs. "It runs" means a contributor (or an agentic loop) can invoke a single command and watch the feature do its work.
- **How to evaluate it.** An automated check — metric, golden output, regression dataset, or an explicit assertion against the spec's Success Criteria — that converts "did this work?" into a yes/no verdict readable by a reviewer or by an LLM driving the loop. For data and ML features, this MUST be a runnable script that reports the spec's success-criteria metrics on a small fixture. For tools and pipelines, this MUST be a quickstart that runs the feature on a checked-in input and asserts on the output.

The harness and evaluation MUST exist *before* the feature code is written, and MUST live in the same PR (or an earlier PR explicitly referenced by the feature PR). This is the system-level companion to Principle I: TDD pins down units; evaluation-first pins down behaviour at the integration / output boundary. The "Independent Test" field on every user story in `spec-template.md` is interpreted from now on as a runnable verification, not just prose.

Trivial changes — typo fixes, documentation-only edits, mechanical refactors with no behavioural delta, and one-line bug fixes covered by an existing regression test — are exempt.

**Rationale:** Without a runnable harness and an explicit evaluation, iteration degrades into "stare at code and hope". Agentic development loops magnify the cost of that hope: an agent that can write ten attempts in five minutes still needs a way to read off which one is correct, and on which axis it improved. Building the harness first inverts the cost: once it exists, every later iteration is cheap to verify, and the loop — human or agentic — converges on the intended behaviour instead of drifting.

### V. Clear, Audience-Aware Writing

All writing in this repository — specs, plans, READMEs, comments, commit messages, PR descriptions, and the constitution itself — MUST target a reader with a computer-science bachelor's-level background who is *not* a working ML researcher. The rules:

- **Gloss ML jargon on first use.** Acronyms, metrics, and algorithms (HCQT, COnPOff F1, NSF vocoder, MFA forced alignment, RVC, and similar) get a one-sentence definition the first time they appear in a document. Use the term freely after that.
- **Cite once.** When a claim depends on a paper, dataset, or repo, link it once near the claim. Don't re-cite for every mention.
- **Shortest version that doesn't lose meaning.** Cut hedges, restatements, scene-setting, and meta-narration. Lists beat paragraphs when items are independent.
- **Specific beats abstract.** Concrete numbers, exact file paths, exact commands, minimal reproducible examples beat generalities.
- **No marketing language.** Forbidden: "robust", "scalable", "world-class", "best-in-class", "leveraging", "synergies", and similar filler.

This principle is forward-looking: documents authored before this amendment are not retroactively rewritten by it. New documents and amended sections MUST follow it.

**Rationale:** The project sits at the intersection of audio ML and software engineering. The people who will actually maintain it — teammates, future-me, code reviewers — have CS backgrounds but won't necessarily have read the singing-transcription literature. Writing for an imaginary ML-researcher peer locks them out; padding to look thorough wastes their time. The rule cuts both ways: explain enough to be readable, and don't say more than you need.

## Commit & Attribution Discipline

Commit messages and pull-request descriptions in this repository MUST NOT attribute authorship to Claude Code, Claude, or any other AI assistant. Specifically, the following are forbidden:

- The `Co-Authored-By: Claude ...` trailer (or any variant naming an AI assistant).
- The "🤖 Generated with [Claude Code]" tagline or any equivalent in PR bodies.
- Any other marketing-style attribution to an AI tool inside commit metadata or PR descriptions.

This rule overrides the default trailer/tagline behaviour that AI development tools may insert automatically. Commit messages SHOULD focus on the *why* of the change, in the style of the existing repository history. PR descriptions SHOULD contain a short summary and a verifiable test plan and nothing more.

**Rationale:** The repository's commit history and PR record are authored by the humans who made the technical decisions; mechanical AI attribution dilutes that signal without adding accountability. Removing it keeps the history readable and keeps responsibility unambiguous.

## Development Workflow & Quality Gates

The following gates apply to every PR before it is mergeable:

1. **Failing test first.** The diff MUST include at least one test that was committed in a state where it failed against the prior code, then passed under the new code. Reviewers MAY ask the author to demonstrate the failing state.
2. **Lint and format clean.** `lint` and `format` (or their language-specific equivalents named in this repo) MUST exit zero with no warnings.
3. **Docs and code move together.** If the diff touches a public-facing surface (CLI flag, config key, file format, exported function/API, or a documented invariant), the same diff MUST update the corresponding docs.
4. **Constitution alignment.** Any PR that introduces a violation of the principles above MUST either fix the violation or include a Complexity Tracking entry in the plan (per `plan-template.md`) with explicit justification and a remediation owner.
5. **Evaluation harness present.** Non-trivial feature PRs MUST check in a runnable harness (single-command invocation) and a runnable evaluation (single-command verdict against the spec's Success Criteria). Reviewers MUST run the evaluation locally or in CI before approving. Trivial PRs as defined in Principle IV are exempt.
6. **Writing readable.** Reviewers MUST flag prose that violates Principle V — undefined ML jargon, padding, marketing language — and the author MUST fix it before merge.

`/speckit-plan` MUST run its Constitution Check against this file and MUST capture, inside Technical Context or Project Structure, the feature's *Evaluation strategy* — the command(s) that run the feature, the command(s) that evaluate it, and the pass/fail criterion referenced back to the spec's Success Criteria. `/speckit-tasks` MUST emit (a) test tasks for every user story (Principle I supersedes the template's "tests are OPTIONAL" note) and (b) at least one evaluation-harness task per user story, ordered before the implementation tasks for that story. `/speckit-implement` MUST refuse to mark an implementation task complete while lint, test, or evaluation gates fail.

## Governance

This constitution is the authoritative source of project-wide rules. Where a template, README, or per-feature plan disagrees with this document, this document wins, and the conflicting file MUST be updated in the same PR.

**Amendment procedure.** Amendments are made by editing this file via `/speckit-constitution`, which records a Sync Impact Report at the top of the file. Every amendment PR MUST update the version line and the `Last Amended` date below.

**Versioning policy.** The version follows semantic versioning:

- **MAJOR** — Backward-incompatible governance changes; a principle is removed, a principle's meaning is materially reversed, or a non-negotiable rule is downgraded.
- **MINOR** — A new principle or a new section is added, or existing guidance is materially expanded.
- **PATCH** — Wording clarifications, typo fixes, non-semantic edits.

**Compliance review.** Compliance is checked on every PR via the gates in *Development Workflow & Quality Gates*. The constitution itself is reviewed at least once per release cycle; the reviewer confirms that the principles still reflect how the team actually wants to work and proposes amendments if not.

**Version**: 1.2.0 | **Ratified**: 2026-06-27 | **Last Amended**: 2026-06-27
