"""Tests for binary mask morphology."""

from __future__ import annotations

import numpy as np

from igcd.morphology import clean_binary_mask, keep_large_components


def test_keep_large_components_removes_small_island() -> None:
    mask = np.zeros((8, 8), dtype="uint8")
    mask[1:5, 1:5] = 1
    mask[7, 7] = 1
    result = keep_large_components(mask, min_area=4)
    assert result.sum() == 16
    assert not result[7, 7]


def test_clean_binary_mask_fills_hole() -> None:
    mask = np.ones((7, 7), dtype="uint8")
    mask[3, 3] = 0
    result = clean_binary_mask(
        mask,
        opening_radius=0,
        closing_radius=0,
        min_object_size=0,
        min_component_area=0,
        fill_holes=True,
    )
    assert result[3, 3] == 1

