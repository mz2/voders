"""``voders eval`` — Success-Criteria pass/fail table (contract: contracts/cli.md, T022)."""

from __future__ import annotations

from voders.evaluation import evaluate, format_report


def eval_command(manifest_path: str, *, strict: bool = False) -> int:
    report = evaluate(manifest_path)
    print(format_report(report))
    if not report.passed:
        return 1
    if strict and any(not c.passed for c in report.criteria):
        return 1
    return 0
