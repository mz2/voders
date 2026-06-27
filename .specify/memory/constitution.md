<!--
Sync Impact Report
==================
Version change: (none) → 1.0.0
Bump rationale: Initial ratification of the project constitution. No prior version exists in the repository (the file held only template placeholders).

Modified principles:
- (initial)

Added sections:
- Core Principles: I. Red-Green-Refactor TDD (NON-NEGOTIABLE), II. Zero-Warning Linting (NON-NEGOTIABLE), III. Documentation Stays Current with the Repo
- Commit & Attribution Discipline
- Development Workflow & Quality Gates
- Governance

Removed sections:
- All `[PLACEHOLDER]` tokens from the template

Templates requiring updates:
- ✅ `.specify/memory/constitution.md` — written
- ✅ `.specify/templates/plan-template.md` — Constitution Check section will be exercised by `/speckit-plan` against the principles below; no template edit required because the section already references the constitution file by design
- ✅ `.specify/templates/spec-template.md` — no edit required; spec template does not encode language about TDD/linting/docs/attribution and the spec workflow itself is unaffected by these principles
- ✅ `.specify/templates/tasks-template.md` — no edit required; the template's note that "tests are OPTIONAL — only include them if explicitly requested in the feature specification" is overridden at execution time by Principle I (TDD is non-negotiable) when `/speckit-tasks` runs against this constitution. Recording this here so `/speckit-tasks` callers know the constitution wins over the template comment.
- ✅ `CLAUDE.md` — no edit required; CLAUDE.md currently delegates context to the active plan and contains no rules in conflict with this constitution

Follow-up TODOs:
- (none)
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

`/speckit-plan` MUST run its Constitution Check against this file. `/speckit-tasks` MUST emit test tasks for every user story (Principle I supersedes the template's "tests are OPTIONAL" note). `/speckit-implement` MUST refuse to mark an implementation task complete while lint or test gates fail.

## Governance

This constitution is the authoritative source of project-wide rules. Where a template, README, or per-feature plan disagrees with this document, this document wins, and the conflicting file MUST be updated in the same PR.

**Amendment procedure.** Amendments are made by editing this file via `/speckit-constitution`, which records a Sync Impact Report at the top of the file. Every amendment PR MUST update the version line and the `Last Amended` date below.

**Versioning policy.** The version follows semantic versioning:

- **MAJOR** — Backward-incompatible governance changes; a principle is removed, a principle's meaning is materially reversed, or a non-negotiable rule is downgraded.
- **MINOR** — A new principle or a new section is added, or existing guidance is materially expanded.
- **PATCH** — Wording clarifications, typo fixes, non-semantic edits.

**Compliance review.** Compliance is checked on every PR via the gates in *Development Workflow & Quality Gates*. The constitution itself is reviewed at least once per release cycle; the reviewer confirms that the principles still reflect how the team actually wants to work and proposes amendments if not.

**Version**: 1.0.0 | **Ratified**: 2026-06-27 | **Last Amended**: 2026-06-27
