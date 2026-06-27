"""``voders eval`` — Success-Criteria pass/fail table (contract: contracts/cli.md, T022)."""

from __future__ import annotations

from voders.evaluation import evaluate, evaluate_score_aug, format_report


def eval_command(manifest_path: str, *, strict: bool = False, suite: str | None = None) -> int:
    if suite == "score_aug":
        report = evaluate_score_aug(manifest_path)
    elif suite in (None, "default"):
        report = evaluate(manifest_path)
    else:
        raise SystemExit(f"unknown eval suite {suite!r}; expected 'default' or 'score_aug'")
    print(format_report(report))
    if not report.passed:
        return 1
    if strict and any(not c.passed for c in report.criteria):
        return 1
    return 0
