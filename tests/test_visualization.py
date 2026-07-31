"""Tests for change-map visualization helpers."""

from __future__ import annotations

import numpy as np

from igcd.visualization import (
    blend_change_overlay,
    change_class_counts,
    change_mask_to_rgb,
)


def test_change_mask_to_rgb_shape_and_colors() -> None:
    mask = np.array([[0, 1], [2, 3]], dtype="uint8")
    rgb = change_mask_to_rgb(mask)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb[1, 0].tolist() == [255, 80, 80]


def test_blend_change_overlay() -> None:
    base = np.zeros((2, 2, 3), dtype="float32")
    mask = np.array([[0, 1], [2, 3]], dtype="uint8")
    overlay = blend_change_overlay(base, mask, alpha=0.5)
    assert overlay.shape == base.shape
    assert overlay[0, 0].sum() == 0
    assert overlay[1, 0, 0] > 0


def test_change_class_counts() -> None:
    mask = np.array([[0, 1], [2, 3]], dtype="uint8")
    counts = change_class_counts(mask)
    assert counts["Background"] == 1
    assert counts["Stable Glacier"] == 1
    assert counts["Glacier Retreat"] == 1
    assert counts["Glacier Advance"] == 1
