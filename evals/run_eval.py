"""Runnable eval harness (Constitution Principle IV).

Thin wrapper around :mod:`voders.evaluation` so the harness is invokable as a single command::

    uv run python evals/run_eval.py --manifest out/smoke/manifest.jsonl

Exits non-zero if any gated Success Criterion fails. The same logic backs ``voders eval``.
"""

from __future__ import annotations

import argparse
import sys

from voders.evaluation import evaluate, format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="voders eval harness")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate(args.manifest)
    print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
