"""US3 integration: per-note volume / dynamics variation (T021, SC-004, FR-008)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voders.audio import read_wav
from voders.config.models import ScoreAugmentationProfile, VolumeKnob
from voders.constants import SAMPLE_RATE
from voders.scores.parse import parse_tsv

from .scoreaug_util import run_score_aug, write_scores


def _per_note_rms(audio: np.ndarray, notes) -> list[float]:
    out = []
    for n in notes:
        a = int(n.onset_s * SAMPLE_RATE)
        b = min(int(n.offset_s * SAMPLE_RATE), audio.size)
        seg = audio[a:b]
        out.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0)
    return out


def test_volume_widens_levels_but_keeps_labels(tmp_path: Path, donor_ah) -> None:
    scores = tmp_path / "scores"
    write_scores(scores, ["score_000"])
    out = tmp_path / "out"
    records = run_score_aug(
        out,
        scores,
        donor_ah,
        ScoreAugmentationProfile(profile_id="vp", volume=VolumeKnob(gain_db_range=(-6.0, 6.0))),
    )

    base = next(r for r in records if r.base_score_id is None and r.score_id == "score_000")
    variant = next(r for r in records if r.score_aug_axis == "volume" and r.score_path)

    assert "augmented/volume/" in variant.score_path
    assert variant.dynamics_applied is True

    # Labels (onset, offset, pitch) byte-identical to the un-varied render.
    bn = parse_tsv(out / base.score_path).score.notes
    vn = parse_tsv(out / variant.score_path).score.notes
    assert [(n.onset_s, n.offset_s, n.pitch_midi) for n in vn] == [
        (n.onset_s, n.offset_s, n.pitch_midi) for n in bn
    ]

    # Rendered per-note level distribution is measurably wider than the un-varied baseline.
    base_audio, _ = read_wav(out / base.audio_path)
    var_audio, _ = read_wav(out / variant.audio_path)
    assert np.std(_per_note_rms(var_audio, vn)) > np.std(_per_note_rms(base_audio, bn))
