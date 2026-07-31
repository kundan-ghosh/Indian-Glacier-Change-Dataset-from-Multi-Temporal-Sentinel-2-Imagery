"""Tests for Drive export organization."""

from __future__ import annotations

from pathlib import Path

from igcd.config import IGCDConfig
from igcd.drive_sync import organize_drive_exports


def test_organize_drive_exports(tmp_path: Path) -> None:
    source = tmp_path / "drive"
    source.mkdir()
    (source / "IGCD_000001_2018_sentinel.tif").write_bytes(b"sentinel")
    (source / "IGCD_000001_2018_mask.tif").write_bytes(b"mask")
    config = IGCDConfig(
        raw={
            "paths": {
                "exports": "exports",
                "processed": "data/processed",
            }
        },
        project_root=tmp_path,
    )

    outputs = organize_drive_exports(source, config)

    assert len(outputs) == 2
    assert (tmp_path / "exports/2018/IGCD_000001_2018_sentinel.tif").exists()
    assert (
        tmp_path / "data/processed/masks/2018/IGCD_000001_2018_mask.tif"
    ).exists()

