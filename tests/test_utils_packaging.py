"""Tests for utility and packaging helpers."""

from __future__ import annotations

import pytest

from igcd.packaging import assign_group_splits
from igcd.utils import chunked, stable_id


def test_chunked() -> None:
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunked_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        list(chunked([1], 0))


def test_stable_id_is_deterministic() -> None:
    assert stable_id("IGCD", ["a", 1]) == stable_id("IGCD", ["a", 1])


def test_assign_group_splits_keeps_unique_glaciers() -> None:
    splits = assign_group_splits(
        ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10"],
        {"train": 0.7, "validation": 0.1, "test": 0.2},
        seed=1,
    )
    assert splits["glacier_id"].is_unique
    assert set(splits["split"]) == {"train", "validation", "test"}

