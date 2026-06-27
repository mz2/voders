"""Resume / checkpoint integration test (T050, SC-011).

A re-run with ``resume=True`` over the same output root must skip already-completed samples:
no new manifest lines, and the checkpoint covers every produced sample.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from voders.config.models import LaneToggle, RunConfig
from voders.corpus.orchestrator import Orchestrator
from voders.corpus.store import CorpusStore
from voders.render.registry import build_lanes

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"


def _config(tmp_path: Path, voice) -> RunConfig:
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    # Two scores so there is more than one checkpointed sample to skip.
    for name in ("score_000.tsv", "score_001.tsv"):
        shutil.copy(SCORES_DIR / name, scores_dir / name)
    return RunConfig(
        run_id="resume",
        master_seed=20260627,
        scores=str(scores_dir / "*.tsv"),
        output_root=str(tmp_path / "out"),
        voices=[voice],
        lanes={"deterministic": LaneToggle(enabled=True)},
    )


def _manifest_lines(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def test_resume_skips_completed_samples(tmp_path: Path, donor_ah) -> None:
    config = _config(tmp_path, donor_ah)

    first = Orchestrator(config, build_lanes(config)).run()
    manifest_path = Path(config.output_root) / "manifest.jsonl"
    lines_after_first = _manifest_lines(manifest_path)
    assert first.attempted == 2
    assert lines_after_first == 2

    completed = CorpusStore(config.output_root).completed_ids()
    assert len(completed) == 2

    # Second run with resume=True: every sample is already completed, so nothing new is produced.
    second = Orchestrator(config, build_lanes(config)).run(resume=True)
    lines_after_resume = _manifest_lines(manifest_path)

    assert second.attempted == 0
    assert lines_after_resume == lines_after_first  # no duplicate manifest lines

    # The checkpoint still covers every sample (no growth, no loss).
    assert CorpusStore(config.output_root).completed_ids() == completed
