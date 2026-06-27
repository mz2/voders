"""Unit tests for the generated-lyric cache-as-artifact (FR-012, SC-007)."""

from __future__ import annotations

from pathlib import Path

from voders.lyrics.cache import LyricCache, request_key


def test_request_key_is_deterministic_and_input_sensitive():
    a = request_key("winter", "m1", "s1", 7, 4)
    assert a == request_key("winter", "m1", "s1", 7, 4)
    assert a != request_key("summer", "m1", "s1", 7, 4)
    assert a != request_key("winter", "m1", "s1", 8, 4)


def test_cache_store_then_load_roundtrips(tmp_path: Path):
    cache = LyricCache(tmp_path, "lyrics")
    key = request_key("winter", "m1", "s1", 7, 3)
    assert cache.load(key) is None  # miss before store
    cache.store(key, "s1", ["win", "ter", None], source="generated", model_id="m1")
    assert cache.load(key) == ["win", "ter", None]
    assert cache.path_for(key).exists()
