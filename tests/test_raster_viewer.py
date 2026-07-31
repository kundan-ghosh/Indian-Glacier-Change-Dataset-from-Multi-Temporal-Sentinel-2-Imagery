"""Tests for raster viewer display math."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from igcd.raster_viewer import (
    RasterPreview,
    default_rgb_bands,
    overlay_mask,
    rgb_composite,
    single_band_display,
    stretch_band,
    threshold_mask,
)


def _preview() -> RasterPreview:
    data = np.stack(
        [
            np.arange(16, dtype="float32").reshape(4, 4),
            np.ones((4, 4), dtype="float32") * 5,
            np.flipud(np.arange(16, dtype="float32").reshape(4, 4)),
        ]
    )
    return RasterPreview(
        path=Path("test.tif"),
        data=data,
        metadata={},
        band_labels=["Band 1", "Band 2", "Band 3"],
    )


def test_stretch_band_returns_unit_range() -> None:
    stretched = stretch_band(np.arange(10, dtype="float32"))
    assert stretched.min() >= 0
    assert stretched.max() <= 1


def test_rgb_composite_shape() -> None:
    rgb = rgb_composite(_preview(), 1, 2, 3)
    assert rgb.shape == (4, 4, 3)


def test_single_band_and_overlay() -> None:
    preview = _preview()
    single = single_band_display(preview, 1)
    mask = threshold_mask(preview, 1, 8)
    overlaid = overlay_mask(rgb_composite(preview, 1, 2, 3), mask)
    assert single.shape == (4, 4)
    assert overlaid.shape == (4, 4, 3)
    assert mask.sum() == 8


def test_default_rgb_bands() -> None:
    assert default_rgb_bands(4) == (4, 3, 2)
    assert default_rgb_bands(3) == (3, 2, 1)
    assert default_rgb_bands(1) == (1, 1, 1)

