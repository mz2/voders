"""Unit tests for the procedural melody generator (score/label coverage axis).

The generator must be deterministic, singable (in-range, monophonic, diatonic), and diverse across
seeds — the properties that make it a useful, validator-friendly coverage source.
"""

from __future__ import annotations

from voders.scores.melody import _MODES, generate_melody


def test_deterministic_in_seed() -> None:
    a = generate_melody(7, score_id="m")
    b = generate_melody(7, score_id="m")
    assert [(n.onset_s, n.offset_s, n.pitch_midi) for n in a.notes] == [
        (n.onset_s, n.offset_s, n.pitch_midi) for n in b.notes
    ]


def test_in_range_and_monophonic() -> None:
    for seed in range(50):
        s = generate_melody(seed, score_id=f"m{seed}", lo=55, hi=79)
        assert s.notes, "melody must be non-empty"
        assert all(55 <= n.pitch_midi <= 79 for n in s.notes)
        assert s.is_monophonic()


def test_note_count_within_bounds() -> None:
    for seed in range(30):
        s = generate_melody(seed, score_id="m", min_notes=6, max_notes=16)
        assert 6 <= len(s.notes) <= 16


def test_pitches_are_diatonic() -> None:
    """Every pitch belongs to the single mode/tonic the melody declares in its source tag."""
    s = generate_melody(3, score_id="m")
    mode = s.source.split(":")[-1]
    if mode in _MODES:  # chromatic fallback only triggers for pathological tessituras
        offsets = set(_MODES[mode])
        # Find a tonic pitch class consistent with all notes belonging to the mode.
        assert any(
            all((n.pitch_midi - tonic) % 12 in offsets for n in s.notes) for tonic in range(12)
        )


def test_diverse_across_seeds() -> None:
    sigs = {tuple((n.pitch_midi, round(n.offset_s - n.onset_s, 3)) for n in generate_melody(
        s, score_id="m").notes) for s in range(40)}
    assert len(sigs) >= 35  # near-unique melodies, not a handful of repeats


def test_cadence_resolves_to_tonic() -> None:
    s = generate_melody(11, score_id="m")
    mode = s.source.split(":")[-1]
    if mode in _MODES:
        last = s.notes[-1].pitch_midi
        # The final pitch is a tonic of some valid key for which the whole melody is diatonic.
        offsets = set(_MODES[mode])
        tonics = [t for t in range(12) if all((n.pitch_midi - t) % 12 in offsets for n in s.notes)]
        assert any(last % 12 == t for t in tonics)
