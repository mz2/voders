"""Unit tests for the learned ``min_note_ms`` threshold (T048, FR-018, FR-019)."""

from __future__ import annotations

from voders.constants import ONSET_TOLERANCE_MS
from voders.scores.analyze import learn_min_note_ms
from voders.scores.models import Note, Score


def _score(durations_s: list[float]) -> Score:
    notes = [Note(onset_s=i, offset_s=i + d, pitch_midi=60) for i, d in enumerate(durations_s)]
    return Score(score_id="s", source="test", notes=notes)


def test_returns_max_of_percentile_hop_and_floor() -> None:
    """min_note_ms = max(data percentile, largest frame hop, 50ms onset tolerance)."""
    # Comfortable durations (300 ms): the data percentile dominates the 50 ms floor.
    learned = learn_min_note_ms([_score([0.3, 0.3, 0.3])])
    assert learned.min_note_ms == learned.data_percentile_ms
    assert learned.min_note_ms >= ONSET_TOLERANCE_MS
    assert learned.onset_tolerance_ms == ONSET_TOLERANCE_MS


def test_all_equal_durations_percentile_equals_duration() -> None:
    learned = learn_min_note_ms([_score([0.25, 0.25, 0.25, 0.25])])
    assert learned.data_percentile_ms == 250.0
    assert learned.min_note_ms == 250.0


def test_floors_at_50ms_when_durations_tiny() -> None:
    """Tiny durations below 50 ms are floored at the onset tolerance (FR-019)."""
    learned = learn_min_note_ms([_score([0.005, 0.006, 0.007])])
    assert learned.data_percentile_ms < ONSET_TOLERANCE_MS
    assert learned.min_note_ms == ONSET_TOLERANCE_MS


def test_method_frame_hop_raises_floor() -> None:
    """A large active frame hop raises the floor above the data percentile and 50ms."""
    learned = learn_min_note_ms(
        [_score([0.02, 0.02, 0.02])],
        method_frame_hops_ms={"slow_method": 80.0},
    )
    assert learned.largest_frame_hop_ms == 80.0
    assert learned.min_note_ms == 80.0
    assert learned.method_frame_hops_ms == {"slow_method": 80.0}


def test_empty_scores_floor_at_onset_tolerance() -> None:
    learned = learn_min_note_ms([])
    assert learned.n_notes == 0
    assert learned.min_note_ms == ONSET_TOLERANCE_MS


def test_to_dict_exposes_contributors() -> None:
    learned = learn_min_note_ms([_score([0.3, 0.3])])
    d = learned.to_dict()
    assert set(d) >= {
        "min_note_ms",
        "data_percentile_ms",
        "largest_frame_hop_ms",
        "onset_tolerance_ms",
        "n_notes",
        "method_frame_hops_ms",
    }
