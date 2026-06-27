"""Validator f0 device selection (CPU / CUDA / Intel XPU / DirectML)."""

from __future__ import annotations

from voders.validate.validator import _select_torch_device


def test_explicit_device_is_passed_through_without_importing_torch():
    # An explicit preference returns as-is and must not require torch to be installed.
    assert _select_torch_device("cpu") == "cpu"
    assert _select_torch_device("cuda") == "cuda"
    assert _select_torch_device("xpu") == "xpu"
    assert _select_torch_device("dml") == "dml"
