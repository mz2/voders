"""voders CLI entry point: ``run``, ``eval``, ``audit``, ``stats`` (contract: contracts/cli.md).

Single non-interactive, config-driven entry point for reproducibility. Exit code 0 on success,
non-zero on a gated failure.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voders", description="Synthetic singing corpus generator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="render a corpus from a Run Config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--output-root", default=None)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--lanes", default=None, help="comma-list overriding enabled lanes")

    p_eval = sub.add_parser("eval", help="validate a corpus and print a Success-Criteria table")
    p_eval.add_argument("--manifest", required=True)
    p_eval.add_argument("--strict", action="store_true")

    p_audit = sub.add_parser("audit", help="license/consent audit over the manifest")
    p_audit.add_argument("--manifest", required=True)

    p_stats = sub.add_parser("stats", help="emit aggregate corpus statistics")
    p_stats.add_argument("--manifest", required=True)
    p_stats.add_argument("--out", default=None)

    p_splits = sub.add_parser("splits", help="write source-stratified train/val splits (issue #7)")
    p_splits.add_argument("--manifest", required=True)
    p_splits.add_argument("--val-fraction", type=float, default=0.2)
    p_splits.add_argument("--seed", type=int, default=0)
    p_splits.add_argument(
        "--stratify",
        default="lane,voice_id,augmentation_profile",
        help="comma-list of provenance fields to stratify by",
    )
    p_splits.add_argument("--out", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        from voders.cli.run import run_command

        lanes = args.lanes.split(",") if args.lanes else None
        return run_command(
            args.config, output_root=args.output_root, resume=args.resume, lanes_override=lanes
        )
    if args.command == "eval":
        from voders.cli.eval import eval_command

        return eval_command(args.manifest, strict=args.strict)
    if args.command == "audit":
        from voders.cli.audit import audit_command

        return audit_command(args.manifest)
    if args.command == "stats":
        from voders.cli.stats import stats_command

        return stats_command(args.manifest, out=args.out)
    if args.command == "splits":
        from voders.cli.splits import splits_command

        stratify = tuple(s.strip() for s in args.stratify.split(",") if s.strip())
        return splits_command(
            args.manifest,
            out=args.out,
            val_fraction=args.val_fraction,
            seed=args.seed,
            stratify=stratify,
        )
    return 2  # pragma: no cover - argparse requires a subcommand


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
