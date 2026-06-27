"""External vocal-melody MIDI as a score-coverage source (issue #9).

Ingest arbitrary MIDI into the same monophonic ``(onset_s, offset_s, pitch_midi)`` Score the rest of
the pipeline consumes — but *carefully*, because same-format is not same-domain. Most public MIDI is
polyphonic, instrumental and not voice-shaped, so this module:

1. **Extracts a melody** — an explicitly named melody/vocal/lead track when present, else a
   **skyline** (the top monophonic line) across the non-drum instruments, else rejects.
2. **Reduces to monophony** — overlaps are resolved in favour of the higher note (skyline).
3. **Makes it singable** — octave-centres the line into the tessitura ``[lo, hi]`` and drops notes
   still out of range; drops sub-``min_note_s`` fragments.
4. **Reports yield + bias** — :class:`ExtractionStats` records the method, in/out counts, drops and
   pre/post pitch range so the silent-filtering bias (#9) is visible, not assumed away.

Licensing/provenance is attached by the importer CLI via a ``_source.json`` sidecar; this module is
pure (no I/O beyond reading the MIDI), so it is unit-testable on synthesised MIDI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from voders.scores.models import Note, Score

_MELODY_NAMES = ("melody", "vocal", "vocals", "lead", "voice", "sing")


@dataclass
class ExtractionStats:
    """Per-file ingest accounting — the visibility #9 asks for (yield + distribution shift)."""

    source_file: str
    method: str  # "melody_track" | "skyline" | "rejected"
    notes_in: int = 0
    notes_out: int = 0
    dropped_out_of_range: int = 0
    dropped_short: int = 0
    transposed_semitones: int = 0
    pitch_in: tuple[int, int] | None = None  # (lo, hi) before filtering
    pitch_out: tuple[int, int] | None = None  # (lo, hi) after filtering
    reject_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.method != "rejected" and self.notes_out > 0


@dataclass
class _RawNote:
    start: float
    end: float
    pitch: int


@dataclass
class _Picked:
    notes: list[_RawNote] = field(default_factory=list)
    method: str = "rejected"


def _melody_track_notes(pm) -> list[_RawNote] | None:
    """Notes of the first instrument whose name looks like a melody/vocal track, if any."""
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        name = (inst.name or "").strip().lower()
        if any(tag in name for tag in _MELODY_NAMES) and inst.notes:
            return [_RawNote(n.start, n.end, n.pitch) for n in inst.notes]
    return None


def _all_pitched_notes(pm) -> list[_RawNote]:
    return [
        _RawNote(n.start, n.end, n.pitch)
        for inst in pm.instruments
        if not inst.is_drum
        for n in inst.notes
    ]


def _skyline(notes: list[_RawNote]) -> list[_RawNote]:
    """Monophonic top line: on overlap keep the higher note, truncating the earlier one."""
    ordered = sorted(notes, key=lambda n: (n.start, -n.pitch))
    out: list[_RawNote] = []
    for n in ordered:
        if out and n.start < out[-1].end:
            if n.pitch > out[-1].pitch:
                out[-1].end = n.start  # truncate the lower, earlier note
                if out[-1].end - out[-1].start <= 1e-3:
                    out.pop()
            else:
                continue  # lower note hidden under the held higher one
        if not out or n.start >= out[-1].end - 1e-6:
            out.append(_RawNote(max(n.start, out[-1].end if out else n.start), n.end, n.pitch))
    return [n for n in out if n.end > n.start]


def extract_melody(
    midi_path: str | Path,
    *,
    lo: int = 55,
    hi: int = 79,
    min_note_s: float = 0.08,
) -> tuple[Score | None, ExtractionStats]:
    """Extract a singable monophonic melody Score from a MIDI file (issue #9).

    Returns ``(Score | None, ExtractionStats)``; ``Score`` is ``None`` when the file yields no
    usable melody (empty, all-drum, or nothing left in range after filtering).
    """
    import pretty_midi

    p = Path(midi_path)
    stats = ExtractionStats(source_file=p.name, method="rejected")
    try:
        pm = pretty_midi.PrettyMIDI(str(p))
    except Exception as e:  # noqa: BLE001 — malformed MIDI is data, not a bug
        stats.reject_reason = f"unreadable: {type(e).__name__}"
        return None, stats

    picked = _melody_track_notes(pm)
    if picked is not None:
        raw, stats.method = picked, "melody_track"
    else:
        allnotes = _all_pitched_notes(pm)
        if not allnotes:
            stats.reject_reason = "no pitched notes"
            return None, stats
        raw, stats.method = _skyline(allnotes), "skyline"

    raw = _skyline(raw)  # enforce monophony even for a declared melody track
    stats.notes_in = len(raw)
    if not raw:
        stats.method, stats.reject_reason = "rejected", "empty after monophonic reduction"
        return None, stats
    pitches_in = [n.pitch for n in raw]
    stats.pitch_in = (min(pitches_in), max(pitches_in))

    # Octave-centre the line into the tessitura, then drop whatever is still out of range.
    center = (lo + hi) / 2
    median = sorted(pitches_in)[len(pitches_in) // 2]
    shift = 12 * round((center - median) / 12)
    stats.transposed_semitones = shift

    kept: list[_RawNote] = []
    for n in raw:
        pitch = n.pitch + shift
        if not (lo <= pitch <= hi):
            stats.dropped_out_of_range += 1
            continue
        if n.end - n.start < min_note_s:
            stats.dropped_short += 1
            continue
        kept.append(_RawNote(n.start, n.end, pitch))

    if not kept:
        stats.method, stats.reject_reason = "rejected", "nothing in singable range"
        return None, stats

    # Shift the whole line so the first onset sits at ~0.2 s and rebuild as Notes.
    t0 = kept[0].start - 0.2
    notes = [
        Note(
            onset_s=round(n.start - t0, 3),
            offset_s=round(n.end - t0, 3),
            pitch_midi=n.pitch,
        )
        for n in kept
    ]
    stats.notes_out = len(notes)
    out_pitches = [n.pitch_midi for n in notes]
    stats.pitch_out = (min(out_pitches), max(out_pitches))
    score = Score(score_id=p.stem, source=f"midi:{p.stem}", notes=notes)
    return score, stats
