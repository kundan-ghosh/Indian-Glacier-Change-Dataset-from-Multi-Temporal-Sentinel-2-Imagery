"""Glacier change-mask generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
STABLE_GLACIER = 1
RETREAT = 2
ADVANCE = 3


@dataclass(frozen=True)
class ChangeStats:
    """Area and pixel-count summary for a glacier change mask."""

    background_pixels: int
    stable_glacier_pixels: int
    retreat_pixels: int
    advance_pixels: int
    pixel_area_m2: float

    @property
    def retreat_area_m2(self) -> float:
        return self.retreat_pixels * self.pixel_area_m2

    @property
    def advance_area_m2(self) -> float:
        return self.advance_pixels * self.pixel_area_m2


def generate_change_mask(
    baseline_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Create a four-class change mask from binary glacier masks.

    Classes are:
    0 background, 1 stable glacier, 2 glacier retreat, 3 glacier advance.
    """

    baseline = np.asarray(baseline_mask).astype(bool)
    target = np.asarray(target_mask).astype(bool)
    if baseline.shape != target.shape:
        raise ValueError(
            "Baseline and target masks must have identical shapes; got "
            f"{baseline.shape} and {target.shape}."
        )
    change = np.zeros(baseline.shape, dtype="uint8")
    change[baseline & target] = STABLE_GLACIER
    change[baseline & ~target] = RETREAT
    change[~baseline & target] = ADVANCE
    return change


def summarize_change_mask(
    change_mask: np.ndarray,
    pixel_area_m2: float,
) -> ChangeStats:
    """Summarize pixel counts and areas for a change mask."""

    labels, counts = np.unique(change_mask.astype("uint8"), return_counts=True)
    count_map = dict(zip(labels.tolist(), counts.tolist()))
    return ChangeStats(
        background_pixels=int(count_map.get(BACKGROUND, 0)),
        stable_glacier_pixels=int(count_map.get(STABLE_GLACIER, 0)),
        retreat_pixels=int(count_map.get(RETREAT, 0)),
        advance_pixels=int(count_map.get(ADVANCE, 0)),
        pixel_area_m2=float(pixel_area_m2),
    )

