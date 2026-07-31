"""Tests for glacier change-mask generation."""

from __future__ import annotations

import numpy as np
import pytest

from igcd.change_detection import (
    ADVANCE,
    BACKGROUND,
    RETREAT,
    STABLE_GLACIER,
    generate_change_mask,
    summarize_change_mask,
)


def test_generate_change_mask_classes() -> None:
    baseline = np.array([[0, 1], [1, 0]], dtype="uint8")
    target = np.array([[0, 1], [0, 1]], dtype="uint8")
    result = generate_change_mask(baseline, target)
    expected = np.array(
        [[BACKGROUND, STABLE_GLACIER], [RETREAT, ADVANCE]],
        dtype="uint8",
    )
    np.testing.assert_array_equal(result, expected)


def test_generate_change_mask_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        generate_change_mask(np.zeros((2, 2)), np.zeros((2, 3)))


def test_summarize_change_mask() -> None:
    change = np.array([[0, 1], [2, 3]], dtype="uint8")
    stats = summarize_change_mask(change, pixel_area_m2=100)
    assert stats.background_pixels == 1
    assert stats.stable_glacier_pixels == 1
    assert stats.retreat_area_m2 == 100
    assert stats.advance_area_m2 == 100

