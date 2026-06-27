"""Contract tests for accompaniment run-config options.

Contract: contracts/run-config-accompaniment.md. ``parse_accompaniment_options`` validates the
free-form ``accompaniment`` lane options into a typed model, enforcing the free-time/bpm and takes
invariants (FR-005/FR-008).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voders.config.models import AccompanimentOptions, parse_accompaniment_options


def test_empty_options_yield_defaults() -> None:
    """An empty dict parses to the documented defaults."""
    opts = parse_accompaniment_options({})
    assert opts.mode == "lego"
    assert opts.backend == "fake"
    assert opts.free_time is True
    assert opts.bpm is None
    assert opts.takes == 1
    assert "MIT" in opts.license_policy


def test_free_time_with_bpm_is_rejected() -> None:
    """free_time=True together with a non-null bpm raises (FR-005)."""
    with pytest.raises(ValidationError):
        parse_accompaniment_options({"free_time": True, "bpm": 120})


def test_zero_takes_is_rejected() -> None:
    """takes must be >= 1."""
    with pytest.raises(ValidationError):
        parse_accompaniment_options({"takes": 0})


def test_unknown_mode_is_rejected() -> None:
    """mode must be lego|complete."""
    with pytest.raises(ValidationError):
        parse_accompaniment_options({"mode": "bogus"})


def test_full_valid_options_round_trip() -> None:
    """A fully specified Complete config parses and round-trips."""
    raw = {
        "mode": "complete",
        "free_time": False,
        "bpm": 90,
        "takes": 3,
        "target_snr_db": [9, 12],
    }
    opts = parse_accompaniment_options(raw)
    assert opts.mode == "complete"
    assert opts.free_time is False
    assert opts.bpm == 90
    assert opts.takes == 3
    assert opts.target_snr_db == [9, 12]

    restored = AccompanimentOptions.model_validate(opts.model_dump())
    assert restored == opts
